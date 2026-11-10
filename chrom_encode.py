import numpy as np
import config

# ---------------------------------------------------------------------------
# Chromosome encoding: one genome = one full gait cycle
# ---------------------------------------------------------------------------
# A gait cycle is n_frames poses in a row (4 by default). Each frame holds
# 8 legs x 3 joints (coxa, femur, tibia), so one genome has
#   n_frames x 8 x 3 = 4 x 8 x 3 = 96 genes, all joint angles in degrees.
#
# The genes are stored flat, frame after frame:
#   genes  0-23  -> frame 0 (L1 c f t, L2 c f t, ... R1 c f t)
#   genes 24-47  -> frame 1
#   genes 48-71  -> frame 2
#   genes 72-95  -> frame 3
# to_frames() reshapes them to (frame, leg, joint) so gait[t, i] is leg i
# in frame t. The last frame loops back to the first, so the cycle repeats.
assert config.n_frames % 2 == 0, "n_frames must be even (half a cycle has to be a whole frame)"

N_FRAMES = config.n_frames
GENES_PER_FRAME = config.n_legs * config.joints_per_leg     # 24
N_GENES = N_FRAMES * GENES_PER_FRAME                         # 96

# Joint caps from config.py (radians, one frame) repeated for every frame
# and converted to degrees, so every gene has its own lowest/highest value.
LOWER = np.rad2deg(np.tile(config.low, N_FRAMES))
UPPER = np.rad2deg(np.tile(config.high, N_FRAMES))


def generate_genome():
    # Every gene is a random angle between its joint's lowest and highest cap
    return np.random.uniform(LOWER, UPPER)


def generate_population():
    return np.array([generate_genome() for _ in range(config.population_size)])


def to_frames(genome):
    # 96 flat genes -> (frames, legs, joints) = (4, 8, 3)
    return np.asarray(genome).reshape(N_FRAMES, config.n_legs, config.joints_per_leg)


def from_frames(gait):
    # (4, 8, 3) -> 96 flat genes (the inverse of to_frames)
    return np.asarray(gait).flatten()


if __name__ == "__main__":
    print(to_frames(a))
    pop = generate_population() 
    print(f"Saved {pop.shape[0]} chromosomes of {pop.shape[1]} genes ")


