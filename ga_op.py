"""Genetic operators + GA loop for the spider gait (Part 1).

Assumptions (adjust if your chrom_encode differs):
  - genome = flat float array of 96 values in DEGREES
  - layout is frame-major: (frame, leg, joint) -> 4 x 8 x 3, so every
    consecutive triple (coxa, femur, tibia) is one leg in one frame
  - ce.LOWER / ce.UPPER are 96-long arrays of joint caps
  - the fitness file is called fitness.py (change the import below if not)
"""
from importlib.machinery import WindowsRegistryFinder

import numpy as np
import config
import chrom_encode as ce
import fitness as fit

rng = np.random.default_rng()

N_GENES = ce.LOWER.size            # 96
JOINTS = 3                         # coxa, femur, tibia
N_BLOCKS = N_GENES // JOINTS       # 32 (frame, leg) blocks 1 block = [coxa,femur,tibia] = 3 , total 96/3 = 32
SPAN = ce.UPPER - ce.LOWER         # per-gene range -> scales mutation size


# ---------------------------------------------------------------------------
# Initialisation: uniform inside the joint caps
# ---------------------------------------------------------------------------
def random_genome():
    return rng.uniform(ce.LOWER, ce.UPPER)


# ---------------------------------------------------------------------------
# Selection: tournament (k = 3)
# ---------------------------------------------------------------------------
# fitness = 1 / (1 + cost). Early costs are huge (ground/lift/slip are scaled
# by 100^2), so almost every fitness is ~0 and a roulette wheel would pick
# near-uniformly (no pressure); later, one good genome can dominate it
# (premature convergence). Tournament uses only the RANK of fitness, so it is
# unaffected by that scale, and k sets the pressure directly.
def tournament_select(pop, fits, k=3):
    selected_idx = rng.integers(0,len(pop),size=k)

    selected_fitness = fits[selected_idx]

    winner_index = selected_idx[np.argmax(selected_fitness)]
  
    return pop[winner_index].copy()

# if this confuses you again its numpy we could do weird array stuff with it 
# copy() is needed else it might pass on genome itself 



# ---------------------------------------------------------------------------
# Crossover A (default): block-uniform, swaps whole (frame, leg) triples
# ---------------------------------------------------------------------------
# A foot position depends on all 3 joint angles of that leg together, so
# ground_cost / lift_cost / stride_cost are all about the coxa+femur+tibia
# combination. Gene-level uniform crossover would split these triples and
# mostly produce broken legs; swapping whole triples keeps good leg poses.
# pc the chance that crossover happens 
def block_uniform_crossover(p1, p2, pc=0.9):
    c1 = p1.copy()
    c2 = p2.copy()

    if rng.random() < pc : 
        
        for i in range(N_BLOCKS):
            if rng.random() < 0.5 : # only happens have the time  
                start = i * JOINTS # JOINTS = 3 
                end =   start + JOINTS 

                c1[start:end]  = p2[start:end] #replacing block of c1 with p2s one 
                c2[start:end] = p1[start:end]


    return c1, c2



# ---------------------------------------------------------------------------
# Crossover B (for comparison): BLX-alpha blend, per gene
# ---------------------------------------------------------------------------
# Picks each child gene uniformly from the parents' interval extended by
# alpha on each side. Real-coded, so it can create new angle values and
# fine-tune the continuous z / stride targets, which swapping cannot do.
def blx_crossover(p1, p2, alpha=0.3, pc=0.9):
    if rng.random() >= pc:
        return p1.copy(), p2.copy()
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    d = hi - lo
    return (rng.uniform(lo - alpha * d, hi + alpha * d),
            rng.uniform(lo - alpha * d, hi + alpha * d))


# ---------------------------------------------------------------------------
# Mutation: Gaussian, per gene, step scaled to each joint's range
# ---------------------------------------------------------------------------
# pm    = 2 / 96 per gene -> about 2 genes change per child
# sigma = 10% of the joint's range early (exploration), shrinking linearly to
#         1% at the end (fine-tuning). Because it scales with SPAN, a step
#         is proportionally the same for coxa (+-45.8), femur (+-68.8) and
#         tibia (+-85.9).
# clip=False lets limit_cost do its job; set True to hard-clamp to the caps.
def gaussian_mutation(g, gen, max_gen, pm=2 / N_GENES,
                      sigma0=0.10, sigma_end=0.01, clip=False):
    sigma = (sigma0 + (sigma_end - sigma0) * gen / max_gen) * SPAN
    mask = rng.random(N_GENES) < pm
    g = g.copy()
    g[mask] += rng.normal(0.0, 1.0, mask.sum()) * sigma[mask]
    return np.clip(g, ce.LOWER, ce.UPPER) if clip else g


# ---------------------------------------------------------------------------
# GA loop (generational, with elitism)
# ---------------------------------------------------------------------------
def evolve(pop_size=config.population_size, max_gen=config.generations,
           k=3, n_elite=2,
           crossover=block_uniform_crossover, seed=None):
    global rng
    if seed is not None:
        rng = np.random.default_rng(seed)

    pop = np.array([ce.generate_genome() for _ in range(pop_size)])
    best_hist, mean_hist = [], []

    for gen in range(max_gen):
        fits = np.array([fit.fitness(g) for g in pop])
        order = np.argsort(fits)[::-1]
        best_hist.append(fits[order[0]])
        mean_hist.append(fits.mean())

        nxt = [pop[i].copy() for i in order[:n_elite]]        # elitism
        while len(nxt) < pop_size:
            p1 = tournament_select(pop, fits, k)
            p2 = tournament_select(pop, fits, k)
            c1, c2 = crossover(p1, p2)
            nxt.append(gaussian_mutation(c1, gen, max_gen))
            if len(nxt) < pop_size:
                nxt.append(gaussian_mutation(c2, gen, max_gen))
        pop = np.array(nxt)

    fits = np.array([fit.fitness(g) for g in pop])
    return pop[np.argmax(fits)], best_hist, mean_hist


if __name__ == "__main__":
    best, best_hist, mean_hist = evolve(seed=0)
    print("best fitness:", fit.fitness(best))
    print("cost breakdown:",
          dict(limit=fit.limit_cost(best), sym=fit.sym_cost(best),
               ground=fit.ground_cost(best), lift=fit.lift_cost(best),
               stride=fit.stride_cost(best), slip=fit.slip_cost(best)))
