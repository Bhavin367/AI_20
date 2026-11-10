import numpy as np
import chrom_encode as ce
import fitness as fit
import ga_op as ga  

np.set_printoptions(suppress=True, precision=2)


def show_costs(name, genome):
    print(f"\n--- {name} ---")
    print("  limit :", round(fit.limit_cost(genome), 1))
    print("  sym   :", round(fit.sym_cost(genome), 1))
    print("  ground:", round(fit.ground_cost(genome), 1))
    print("  lift  :", round(fit.lift_cost(genome), 1))
    print("  stride:", round(fit.stride_cost(genome), 1))
    print("  slip  :", round(fit.slip_cost(genome), 1))
    print("  fitness:", fit.fitness(genome))


def show_walk(genome):
    feet = fit.foot_positions(genome)
    print("foot height per frame (rows = frames, cols = L1 L2 L3 L4 R4 R3 R2 R1):")
    print(feet[..., 2])
    print("swinging legs (1 = in the air):")
    print(fit.SWING.astype(int))
    print("stride per leg:", fit.stride_length(genome))


ge = ce.generate_genome()
print("genes:", ge.size, f"({ce.N_FRAMES} frames x {ce.GENES_PER_FRAME})")
print("frame 0:\n", ce.to_frames(ge)[0])
print("caps low :", fit.LOWER[:3])
print("caps high:", fit.UPPER[:3])

show_costs("random gait", ge)
show_walk(ge)

stand = np.tile([0, -40, -40], ce.N_FRAMES * 8).astype(float)
show_costs("standing still", stand)

best, best_hist, mean_hist = ga.evolve(seed=0)

show_costs("GA best", best)
show_walk(best)
