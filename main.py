import numpy as np
import matplotlib.pyplot as plt

import chrom_encode as ce
import fitness as fit
import ga_op as ga            # full 96-gene encoding
import ga_mirror as gm        # mirrored 48-gene encoding (right legs built from left)
import viz

np.set_printoptions(suppress=True, precision=2)

GENERATIONS = 500             # same budget for both encodings so the comparison is fair
POP_SIZE = 100
SEED = 0
MAKE_GIF = True               # set False to skip the (slow-ish) animation


def show_costs(name, genome):
    print(f"\n--- {name} ---")
    print("  limit :", round(fit.limit_cost(genome), 1))
    print("  sym   :", round(fit.sym_cost(genome), 1))
    print("  ground:", round(fit.ground_cost(genome), 1))
    print("  lift  :", round(fit.lift_cost(genome), 1))
    print("  stride:", round(fit.stride_cost(genome), 1))
    print("  slip  :", round(fit.slip_cost(genome), 1))
    print("  total cost:", round(fit.total_cost(genome), 1))
    print("  fitness:", fit.fitness(genome))


def show_walk(genome):
    feet = fit.foot_positions(genome)
    print("foot height per frame (rows = frames, cols = L1 L2 L3 L4 R4 R3 R2 R1):")
    print(feet[..., 2])
    print("swinging legs (1 = in the air):")
    print(fit.SWING.astype(int))
    print("stride per leg:", fit.stride_length(genome))


# --- baselines -------------------------------------------------------------
ge = ce.generate_genome()
print("genes:", ge.size, f"({ce.N_FRAMES} frames x {ce.GENES_PER_FRAME})")
show_costs("random gait", ge)

stand = np.tile([0, -40, -40], ce.N_FRAMES * 8).astype(float)
show_costs("standing still", stand)

# --- sanity check: a decoded mirrored genome must have zero symmetry cost ---
test = gm.decode(gm.random_genome())
print("\nmirror decode check, sym cost (should be 0):", fit.sym_cost(test))

# --- run both GAs ----------------------------------------------------------
print("\n=== full 96-gene encoding ===")
best_full, bh_full, mh_full = ga.evolve(pop_size=POP_SIZE, generations=GENERATIONS, seed=SEED)
show_costs("GA best (96 genes)", best_full)

print("\n=== mirrored 48-gene encoding ===")
best_mirror, bh_mirror, mh_mirror = gm.evolve(pop_size=POP_SIZE, generations=GENERATIONS, seed=SEED)
show_costs("GA best (mirrored 48 genes)", best_mirror)
show_walk(best_mirror)

# --- plots (saved as PNG/GIF, shown together at the end) --------------------
viz.plot_histories({"96 genes": (bh_full, mh_full),
                    "mirrored 48": (bh_mirror, mh_mirror)})
viz.plot_costs({"random": ge, "96 genes": best_full, "mirrored 48": best_mirror})
viz.plot_key_frames(best_mirror, "poses_mirror.png")
viz.plot_key_frames(best_full, "poses_full.png")
viz.export_gait(best_mirror, "gait.csv")
if MAKE_GIF:
    viz.animate(best_mirror, "walk.gif")
    print("saved walk.gif")
print("saved fitness.png, costs.png, poses_*.png, gait.csv")

plt.show()
