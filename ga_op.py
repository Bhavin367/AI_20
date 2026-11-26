"""Genetic operators + GA loop for the spider gait (Part 1).

Assumptions (adjust if your chrom_encode differs):
  - genome = flat float array of 96 values in DEGREES
  - layout is frame-major: (frame, leg, joint) -> 4 x 8 x 3, so every
    consecutive triple (coxa, femur, tibia) is one leg in one frame
  - ce.LOWER / ce.UPPER are 96-long arrays of joint caps
  - the fitness file is called fitness.py (change the import below if not)
"""

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
def tournament_select(pop, fits, k=config.k):
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
def block_uniform_crossover(p1, p2, pc=config.pc):
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
def blx_crossover(p1, p2, alpha=config.blx_alpha, pc=config.pc):
    c1, c2 = p1.copy(), p2.copy()

    if rng.random() < pc:
        lo = np.minimum(p1, p2)
        hi = np.maximum(p1, p2)
        margin = alpha * (hi - lo)

        c1 = rng.uniform(lo - margin, hi + margin)
        c2 = rng.uniform(lo - margin, hi + margin)

    return c1, c2


# ---------------------------------------------------------------------------
# Mutation A (default): per-gene Gaussian, scaled to each gene's range
# ---------------------------------------------------------------------------
# Each gene mutates independently with probability pm. A mutated gene gets
# noise ~ N(0, sigma_frac * SPAN), so the step size is a fraction of that
# gene's allowed range (a joint with a wide cap takes bigger steps than a
# narrow one). Small nudges fine-tune an already-good leg pose, which suits
# real-coded angles. Results are clipped so we never leave the joint caps.
# pm = 0.05 -> about 5 of 96 genes change per child on average.
def gaussian_mutation(genome, pm=config.pm, sigma_frac=config.sigma_fixed):
    child = genome.copy()

    mask = rng.random(N_GENES) < pm                 # which genes mutate
    noise = rng.normal(0.0, sigma_frac * SPAN)      # one noise value per gene
    child[mask] += noise[mask]                      # only apply to masked genes

    return np.clip(child, ce.LOWER, ce.UPPER)


# ---------------------------------------------------------------------------
# Mutation B: block reset, re-randomises a whole (frame, leg) triple
# ---------------------------------------------------------------------------
# Gaussian noise only makes local moves. This makes a big jump on one leg
# pose at a time. It keeps coxa/femur/tibia together (same reasoning as the
# block crossover) and helps escape local optima / restore lost diversity.
# Keep pm_block low, since it is disruptive.
def block_reset_mutation(genome, pm_block=config.pm_block):
    child = genome.copy()

    for i in range(N_BLOCKS):
        if rng.random() < pm_block:
            start = i * JOINTS
            end = start + JOINTS
            child[start:end] = rng.uniform(ce.LOWER[start:end], ce.UPPER[start:end])

    return child


# ---------------------------------------------------------------------------
# Adaptive step size: sigma shrinks linearly over the run
# ---------------------------------------------------------------------------
# Early on we want big steps to explore; late on we want small ones to
# fine-tune. Pass the result in as sigma_frac.
def sigma_schedule(gen, max_gen, start=config.sigma_start, end=config.sigma_end):
    t = gen / max(1, max_gen - 1)
    return start + t * (end - start)


# ---------------------------------------------------------------------------
# Wrapper used by the GA loop
# ---------------------------------------------------------------------------
def mutate(genome, pm=config.pm, sigma_frac=config.sigma_fixed, pm_block=config.pm_block):
    genome = gaussian_mutation(genome, pm, sigma_frac)
    genome = block_reset_mutation(genome, pm_block)
    return genome 


# ---------------------------------------------------------------------------
# The GA loop
# ---------------------------------------------------------------------------
# Returns (best_genome, best_hist, mean_hist) because main.py expects that.
# Every parameter is an argument so it can be tuned/compared in the report.
def evolve(pop_size=config.population_size,
           generations=config.generations,
           pc=config.pc,                 # crossover probability
           pm=config.pm,                # per-gene mutation probability
           pm_block=config.pm_block,          # per-block reset probability
           k=config.k,                    # tournament size
           elite=config.elite,                # best genomes copied unchanged
           crossover=block_uniform_crossover,
           adaptive=True,          # shrink sigma over the run
           sigma_start=config.sigma_start,
           sigma_end=config.sigma_end,
           sigma_fixed=config.sigma_fixed,        # used when adaptive=False
           target=config.target_fitness,           # stop early if fitness reaches this
           seed=None,
           verbose=True):
    global rng
    if seed is not None:
        rng = np.random.default_rng(seed)   # reproducible runs

    # 1. initialise + evaluate
    pop = np.array([random_genome() for _ in range(pop_size)])
    fits = np.array([fit.fitness(g) for g in pop])
    best_hist, mean_hist = [], []

    for gen in range(generations):
        # 2. elitism: keep the best `elite` genomes untouched
        order = np.argsort(fits)[::-1]
        new_pop = [pop[i].copy() for i in order[:elite]]

        sigma = sigma_schedule(gen, generations, sigma_start, sigma_end) if adaptive else sigma_fixed

        # 3. breed until the population is full
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fits, k)
            p2 = tournament_select(pop, fits, k)
            c1, c2 = crossover(p1, p2, pc=pc)
            new_pop.append(mutate(c1, pm, sigma, pm_block))
            if len(new_pop) < pop_size:
                new_pop.append(mutate(c2, pm, sigma, pm_block))

        # 4. evaluate the new generation
        pop = np.array(new_pop)
        fits = np.array([fit.fitness(g) for g in pop])

        # 5. record stats for the plots
        best_hist.append(fits.max())
        mean_hist.append(fits.mean())
        if verbose and gen % 50 == 0:
            costs = 1.0 / fits - 1.0
            print(f"gen {gen:4d}  best cost {costs.min():10.1f}  mean cost {costs.mean():12.1f}")

        # 6. termination: good enough
        if fits.max() >= target:
            break

    return pop[np.argmax(fits)].copy(), best_hist, mean_hist
