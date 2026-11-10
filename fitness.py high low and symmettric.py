import numpy as np
import config

LEG_NAMES = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]


def as_legs(genome):
    # Reshape the 24 genes into 8 rows (one per leg) of [coxa, femur, tibia]
    return np.asarray(genome).reshape(8, 3)


# ---------------------------------------------------------------------------
# Fitness function 1: joint limits (lowest and highest cap)
# ---------------------------------------------------------------------------
# Every joint has a lowest and highest allowed angle, the same for all 8 legs:
#
#   Joint  | Lowest  | Highest | (radians in config.py)
#   coxa   | -45.8   | +45.8   | +-0.8
#   femur  | -68.8   | +68.8   | +-1.2
#   tibia  | -85.9   | +85.9   | +-1.5
#
# The coxa has the smallest range because it swings the whole leg sideways
# from the body; the tibia has the largest because it connects to the foot
# and needs the most freedom to reach the ground.
#
# config.py stores the caps in radians, but genomes are in degrees, so they
# are converted here. Both arrays have 24 values (the 3 caps repeated per leg).
LOWER = np.rad2deg(config.low)
UPPER = np.rad2deg(config.high)


def limit_cost(genome):
    # For each angle, measure how far it is outside its cap (0 if inside):
    #   below the lowest cap  -> LOWER - angle
    #   above the highest cap -> angle - UPPER
    # The distances are squared so large violations are punished much harder
    # than small ones (1 degree over costs 1, 10 degrees over costs 100).
    # A genome with every angle inside its caps has a cost of 0.
    g = np.asarray(genome)
    return np.sum(np.clip(LOWER - g, 0, None) ** 2 + np.clip(g - UPPER, 0, None) ** 2)


# ---------------------------------------------------------------------------
# Fitness function 2: left/right symmetry
# ---------------------------------------------------------------------------
# Each left leg should be a mirror image of its partner on the right side:
#   L1 <-> R1,  L2 <-> R2,  L3 <-> R3,  L4 <-> R4
#
# Mirroring across the body's centre line (the head direction, 0 degrees):
#   - coxa swings the leg sideways, so the mirror angle is the NEGATIVE.
#     Leg direction = base angle + coxa, so L1 at 45 + c mirrors R1 at -45 - c,
#     which means coxa_R = -coxa_L
#   - femur and tibia bend the leg up/down, so the mirror angle is the SAME
#
# MIRROR flips the sign of the coxa and leaves femur and tibia unchanged.
MIRROR = np.array([-1, 1, 1])


def sym_cost(genome):
    # Legs are stored L1 L2 L3 L4 R4 R3 R2 R1:
    #   legs[:4]    -> L1, L2, L3, L4
    #   legs[:3:-1] -> R1, R2, R3, R4 (right side reversed so the pairs line up)
    # Mirror each left leg and subtract its right partner. If the robot is
    # perfectly symmetric every difference is 0. Differences are squared and
    # summed, the same way as limit_cost, so both costs are on the same scale.
    legs = as_legs(genome)
    return np.sum((legs[:4] * MIRROR - legs[:3:-1]) ** 2)


# ---------------------------------------------------------------------------
# Overall fitness
# ---------------------------------------------------------------------------
# Each cost is multiplied by its weight from config.py and added together.
# 1 / (1 + cost) turns the total into a score between 0 and 1:
#   1.0  -> no penalties (inside every cap and perfectly symmetric)
#   ~0   -> large penalties
def fitness(genome):
    cost = config.w_limit * limit_cost(genome) + config.w_sym * sym_cost(genome)
    return 1.0 / (1.0 + cost)
