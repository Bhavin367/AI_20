# Genetic Algorithm

## Full Genome Implementation (ga_op.py)

### Selection: Tournament Selection
**Function:** `tournament_select()`

Three individuals are randomly selected from the population (`k=3`), and the one with the highest fitness is chosen as a parent.

**Justification:** Simple to implement, favours fitter individuals and maintains some diversity through random selection.

### Crossover: Block Uniform Crossover
**Function:** `block_uniform_crossover()`

Uses a crossover probability of 0.9. Each block of three genes (representing one leg's joint angles) has a 50% chance of being exchanged between the parents.

**Justification:** Keeping the three joint angles together helps preserve coordinated leg movements rather than mixing individual joint values randomly.

An alternative `blx_crossover()` is also implemented but is not used by default.

### Mutation: Gaussian and Block Reset
**Functions:** `gaussian_mutation()`, `block_reset_mutation()`, `mutate()`

Two mutation methods are combined:
- **Gaussian mutation:** Each gene has a 5% chance of receiving a random adjustment, with the noise standard deviation set to 10% of its allowed range.
- **Block reset mutation:** Each three-gene block has a 3% chance of being completely randomised within its allowed joint-angle limits.

**Justification:** Gaussian mutation makes small adjustments to existing solutions, while block reset introduces larger changes to encourage exploration and avoid local optima.

The `sigma_schedule()` function gradually reduces mutation strength from 0.15 to 0.02 across generations when adaptive mutation is enabled.

---

## Mirrored Genome Implementation (ga_mirror.py)

### Selection: Tournament Selection
**Function:** `tournament_select()`

Uses the same tournament selection method, randomly selecting three individuals and choosing the one with the highest fitness.

**Justification:** Favours fitter individuals while maintaining population diversity.

### Crossover: Block Uniform Crossover
**Function:** `block_uniform_crossover()`

Uses a crossover probability of 0.9, with each three-gene block having a 50% chance of being exchanged between parents.

**Justification:** Preserves groups of joint angles during reproduction, helping maintain coordinated movements.

An alternative `blx_crossover()` is also implemented but is not used by default.

### Mutation: Gaussian and Block Reset
**Functions:** `gaussian_mutation()`, `block_reset_mutation()`, `mutate()`

Combines two methods:
- **Gaussian mutation:** 5% probability per gene, introducing small random adjustments.
- **Block reset mutation:** 3% probability per block, replacing all three joint angles with random values within their permitted limits.

**Justification:** Combines fine-tuning of existing solutions with larger random changes to explore new configurations.

The `sigma_schedule()` function gradually reduces mutation strength from 0.15 to 0.02 across generations.
