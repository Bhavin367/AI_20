import numpy as np
import matplotlib.pyplot as plt

import config
import ga_op as ga
import ga_mirror as gm

SEEDS = [0, 1, 2, 3, 4]
POP_SIZE = config.ablation_population_size
GENERATIONS = config.ablation_generations


def run_multi_seed(label, fn, **kwargs):
    curves = []
    final_costs = []
    for s in SEEDS:
        _, bh, _ = fn(pop_size=POP_SIZE, generations=GENERATIONS, seed=s, verbose=False, **kwargs)
        costs = 1.0 / np.array(bh) - 1.0
        if len(costs) < GENERATIONS:
            costs = np.pad(costs, (0, GENERATIONS - len(costs)), mode="edge")
        curves.append(costs)
        final_costs.append(costs[-1])
    curves = np.array(curves)
    print(f"{label:<32} mean={np.mean(final_costs):8.1f}  std={np.std(final_costs):7.1f}  min={np.min(final_costs):7.1f}")
    return curves


def main():
    print(f"Running {len(SEEDS)}-seed GA ablations ({POP_SIZE} pop, {GENERATIONS} gens)...")
    configs = [
        ("96 genes (block + adaptive)", ga.evolve, dict(crossover=ga.block_uniform_crossover, adaptive=True)),
        ("48 mirror (block + adaptive)", gm.evolve, dict(crossover=gm.block_uniform_crossover, adaptive=True)),
        ("48 mirror (BLX-a + adaptive)", gm.evolve, dict(crossover=gm.blx_crossover, adaptive=True)),
        ("48 mirror (block + fixed sig)", gm.evolve, dict(crossover=gm.block_uniform_crossover, adaptive=False)),
    ]

    plt.figure(figsize=(8, 5))
    for label, fn, kw in configs:
        curves = run_multi_seed(label, fn, **kw)
        mean_c = curves.mean(axis=0)
        std_c = curves.std(axis=0)
        gens = np.arange(GENERATIONS)
        line, = plt.semilogy(gens, mean_c, label=label)
        plt.fill_between(gens, np.maximum(1.0, mean_c - std_c), mean_c + std_c,
                         color=line.get_color(), alpha=0.15)

    plt.xlabel("Generation")
    plt.ylabel("Best total cost (mean +/- 1 std, log scale)")
    plt.title(f"GA Ablation Study ({len(SEEDS)} seeds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ga_ablation.png", dpi=150)
    print("saved ga_ablation.png")

    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
