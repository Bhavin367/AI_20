"""GA with a MIRRORED (half-size) chromosome for the spider gait.

Idea: sym_cost wants every right leg to be the mirror of its left partner,
half a cycle later. Instead of hoping the GA discovers that, we only evolve
the LEFT legs (4 frames x 4 legs x 3 joints = 48 genes) and BUILD the right
legs from them. Symmetry is then exactly 0 by construction and the search
space is half the size.

Chromosome layout (48 genes, degrees), frame-major:
    (frame, leg, joint) = (4, 4, 3), legs in order L1 L2 L3 L4

decode() expands it to the 96-gene layout used by fitness.py
(L1 L2 L3 L4 R4 R3 R2 R1 in every frame).
"""

import numpy as np
import config
import chrom_encode as ce
import fitness as fit

rng = np.random.default_rng()

JOINTS = 3
N_LEFT = config.n_legs // 2                    # 4 left legs
N_GENES = ce.N_FRAMES * N_LEFT * JOINTS        # 48
N_BLOCKS = N_GENES // JOINTS                   # 16 (frame, leg) blocks

# Caps for the left legs only (same caps for every leg, so just slice).
LOWER = ce.LOWER[:N_GENES]
UPPER = ce.UPPER[:N_GENES]
SPAN = UPPER - LOWER


# ---------------------------------------------------------------------------
# Decoding: 48 evolved genes -> 96-gene genome for fitness.py
# ---------------------------------------------------------------------------
# Right leg at frame t = mirror of the left partner at frame t - HALF_CYCLE.
# (This is exactly what sym_cost checks.) np.roll(x, h)[t] = x[t - h].
# Mirror = negate coxa, keep femur and tibia.
def decode(half):
    left = np.asarray(half).reshape(ce.N_FRAMES, N_LEFT, JOINTS)
    right = np.roll(left * fit.MIRROR, fit.HALF_CYCLE, axis=0)   # R1..R4 order
    full = np.concatenate([left, right[:, ::-1]], axis=1)        # L1-L4, R4-R1
    return full.reshape(-1)


def cost(half):
    return fit.total_cost(decode(half))


def fitness(half):
    return fit.fitness(decode(half))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def random_genome():
    return rng.uniform(LOWER, UPPER)


# ---------------------------------------------------------------------------
# Selection: tournament (same reasoning as ga_op.py - uses rank, not scale)
# ---------------------------------------------------------------------------
def tournament_select(pop, fits, k=config.k):
    idx = rng.integers(0, len(pop), size=k)
    return pop[idx[np.argmax(fits[idx])]].copy()


# ---------------------------------------------------------------------------
# Crossover: block-uniform on (frame, leg) triples
# ---------------------------------------------------------------------------
def block_uniform_crossover(p1, p2, pc=config.pc):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() < pc:
        for i in range(N_BLOCKS):
            if rng.random() < 0.5:
                s, e = i * JOINTS, (i + 1) * JOINTS
                c1[s:e], c2[s:e] = p2[s:e], p1[s:e]
    return c1, c2


def blx_crossover(p1, p2, alpha=config.blx_alpha, pc=config.pc):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() < pc:
        lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
        margin = alpha * (hi - lo)
        c1 = rng.uniform(lo - margin, hi + margin)
        c2 = rng.uniform(lo - margin, hi + margin)
    return np.clip(c1, LOWER, UPPER), np.clip(c2, LOWER, UPPER)


# ---------------------------------------------------------------------------
# Mutation: per-gene Gaussian + rare block reset (as in ga_op.py)
# ---------------------------------------------------------------------------
def gaussian_mutation(genome, pm=config.pm, sigma_frac=config.sigma_fixed):
    child = genome.copy()
    mask = rng.random(N_GENES) < pm
    noise = rng.normal(0.0, sigma_frac * SPAN)
    child[mask] += noise[mask]
    return np.clip(child, LOWER, UPPER)


def block_reset_mutation(genome, pm_block=config.pm_block):
    child = genome.copy()
    for i in range(N_BLOCKS):
        if rng.random() < pm_block:
            s, e = i * JOINTS, (i + 1) * JOINTS
            child[s:e] = rng.uniform(LOWER[s:e], UPPER[s:e])
    return child


def sigma_schedule(gen, max_gen, start=config.sigma_start, end=config.sigma_end):
    return start + gen / max(1, max_gen - 1) * (end - start)


def mutate(genome, pm=config.pm, sigma_frac=config.sigma_fixed, pm_block=config.pm_block):
    return block_reset_mutation(gaussian_mutation(genome, pm, sigma_frac), pm_block)


# ---------------------------------------------------------------------------
# GA loop
# ---------------------------------------------------------------------------
# Returns (best_96_gene_genome, best_hist, mean_hist) so it plugs into the
# same code as ga_op.evolve. Histories are FITNESS values (1/(1+cost)).
def evolve(pop_size=config.population_size,
           generations=config.generations,
           pc=config.pc, pm=config.pm, pm_block=config.pm_block, k=config.k, elite=config.elite,
           crossover=block_uniform_crossover,
           adaptive=True, sigma_start=config.sigma_start, sigma_end=config.sigma_end, sigma_fixed=config.sigma_fixed,
           target=config.target_fitness, seed=None, verbose=True):
    global rng
    if seed is not None:
        rng = np.random.default_rng(seed)

    pop = np.array([random_genome() for _ in range(pop_size)])
    fits = np.array([fitness(g) for g in pop])
    best_hist, mean_hist = [], []

    for gen in range(generations):
        order = np.argsort(fits)[::-1]
        new_pop = [pop[i].copy() for i in order[:elite]]       # elitism
        sigma = sigma_schedule(gen, generations, sigma_start, sigma_end) if adaptive else sigma_fixed

        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fits, k)
            p2 = tournament_select(pop, fits, k)
            c1, c2 = crossover(p1, p2, pc=pc)
            new_pop.append(mutate(c1, pm, sigma, pm_block))
            if len(new_pop) < pop_size:
                new_pop.append(mutate(c2, pm, sigma, pm_block))

        pop = np.array(new_pop)
        fits = np.array([fitness(g) for g in pop])
        best_hist.append(fits.max())
        mean_hist.append(fits.mean())

        if verbose and gen % 50 == 0:
            costs = 1.0 / fits - 1.0
            print(f"[mirror] gen {gen:4d}  best cost {costs.min():10.1f}  mean cost {costs.mean():12.1f}")
        if fits.max() >= target:
            break

    return decode(pop[np.argmax(fits)]), best_hist, mean_hist
