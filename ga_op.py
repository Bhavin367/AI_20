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
    c1, c2 = p1.copy(), p2.copy()

    if rng.random() < pc:
        lo = np.minimum(p1, p2)
        hi = np.maximum(p1, p2)
        margin = alpha * (hi - lo)

        c1 = rng.uniform(lo - margin, hi + margin)
        c2 = rng.uniform(lo - margin, hi + margin)

    return c1, c2
