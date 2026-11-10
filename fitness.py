import numpy as np
import config
import chrom_encode as ce

LEG_NAMES = config.leg_names          # L1 L2 L3 L4 R4 R3 R2 R1
N_FRAMES = ce.N_FRAMES


# ---------------------------------------------------------------------------
# Fitness function 1: joint limits (lowest and highest cap)
# ---------------------------------------------------------------------------
# Every joint has a lowest and highest allowed angle, the same for all 8 legs
# and every frame of the gait:
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
# config.py stores the caps in radians, but genomes are in degrees, so
# chrom_encode converts them and repeats them for every frame (96 values).
LOWER = ce.LOWER
UPPER = ce.UPPER


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
# The walking pattern: alternating tetrapod gait
# ---------------------------------------------------------------------------
# The legs are split into two groups of four that take turns stepping, so
# at least 4 feet are always on the ground and the robot stays balanced:
#   Group A: L1, L3, R4, R2   (even positions in L1 L2 L3 L4 R4 R3 R2 R1)
#   Group B: L2, L4, R3, R1   (odd positions)
# Each group has 2 legs on each side, and every left leg's partner on the
# right (L1<->R1 etc.) is in the other group.
#
# Each leg either SWINGS (foot in the air, moving forward) or is in STANCE
# (foot on the ground, pushing back). Group B does exactly what group A did
# half a cycle earlier. With 4 frames:
#
#   Frame | Group A                      | Group B
#   0     | swing (foot up, forward)     | stance (pushing back)
#   1     | stance (lands at the front)  | stance (lifts off at the back)
#   2     | stance (pushing back)        | swing (foot up, forward)
#   3     | stance (lifts off at back)   | stance (lands at the front)
#
# Frame 3 loops back to frame 0, so the cycle repeats forever. At most one
# group is in the air at a time, so 4-8 feet are always on the ground.
HALF_CYCLE = N_FRAMES // 2
N_SWING = max(1, N_FRAMES // 4)                  # frames each leg spends in the air
GROUP_A = np.arange(config.n_legs) % 2 == 0      # True for L1, L3, R4, R2

swing_a = np.arange(N_FRAMES) < N_SWING          # group A swings in frame 0
swing_b = np.roll(swing_a, HALF_CYCLE)           # group B swings half a cycle later
SWING = np.where(GROUP_A, swing_a[:, None], swing_b[:, None])   # (frames, legs)
STANCE = ~SWING

# First and last stance frame of each leg (where the foot lands and lifts off)
first_a, last_a = N_SWING, N_FRAMES - 1
STANCE_START = np.where(GROUP_A, first_a, (first_a + HALF_CYCLE) % N_FRAMES)
STANCE_END = np.where(GROUP_A, last_a, (last_a + HALF_CYCLE) % N_FRAMES)


# ---------------------------------------------------------------------------
# Fitness function 2: left/right symmetry while walking
# ---------------------------------------------------------------------------
# Because partners are in opposite groups, a right leg should do exactly
# what its left partner did HALF A CYCLE (2 frames) earlier, mirrored:
#   R1 at frame 2 = mirror of L1 at frame 0
#   R1 at frame 3 = mirror of L1 at frame 1   ... and so on, wrapping round.
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
    # Legs are stored L1 L2 L3 L4 R4 R3 R2 R1 in every frame:
    #   gait[:, :4]    -> L1, L2, L3, L4 in every frame
    #   gait[:, :3:-1] -> R1, R2, R3, R4 (right side reversed so pairs line up)
    # np.roll shifts the right legs back by half a cycle, so frame t of the
    # left legs is compared with frame t + 2 of the right legs.
    # If the gait is perfectly symmetric every difference is 0. Differences
    # are squared and summed, the same way as limit_cost.
    gait = ce.to_frames(genome)
    left = gait[:, :4]
    right_later = np.roll(gait[:, :3:-1], -HALF_CYCLE, axis=0)
    return np.sum((left * MIRROR - right_later) ** 2)


# ---------------------------------------------------------------------------
# Forward kinematics: where is each foot?
# ---------------------------------------------------------------------------
# The body sits at height z = 0, head pointing along +x, left side along +y.
# Each leg is mounted on the edge of the body (an ellipse with half-length A
# and half-width B) at its base angle, e.g. L1 at 45 degrees.
#
# From the mount the leg is a chain of 3 segments (coxa 1.2, femur 0.7,
# tibia 1.0). Seen from above, the whole leg points in the direction
#   base angle + coxa angle
# Seen from the side, each segment tilts up/down (positive = up):
#   coxa  tilt = 30 degrees (fixed coxa_elevation)
#   femur tilt = coxa tilt  + femur angle
#   tibia tilt = femur tilt + tibia angle
# Adding up the segments gives how far out the foot reaches and its height.
SEG_COXA, SEG_FEMUR, SEG_TIBIA = config.segment_length
BASE = np.concatenate([config.left_leg_angles, config.right_leg_angles])   # radians
MOUNT_X = config.A * np.cos(BASE)
MOUNT_Y = config.B * np.sin(BASE)


def foot_positions(genome):
    # Returns (frames, legs, 3): the x, y, z of every foot in every frame
    q = np.deg2rad(ce.to_frames(genome))
    heading = BASE + q[..., 0]
    tilt_coxa = config.coxa_elevation
    tilt_femur = tilt_coxa + q[..., 1]
    tilt_tibia = tilt_femur + q[..., 2]

    reach = (SEG_COXA * np.cos(tilt_coxa) + SEG_FEMUR * np.cos(tilt_femur)
             + SEG_TIBIA * np.cos(tilt_tibia))
    z = (SEG_COXA * np.sin(tilt_coxa) + SEG_FEMUR * np.sin(tilt_femur)
         + SEG_TIBIA * np.sin(tilt_tibia))
    x = MOUNT_X + reach * np.cos(heading)
    y = MOUNT_Y + reach * np.sin(heading)
    return np.stack([x, y, z], axis=-1)


# Distances are in body units (around 1) but angles are in degrees (tens),
# so distance errors are scaled by 100: being 0.1 units off costs the same
# as an angle being 10 degrees past its cap (both cost 100).
DIST_SCALE = 100


# ---------------------------------------------------------------------------
# Fitness function 3: feet on the ground during stance
# ---------------------------------------------------------------------------
# While a leg is pushing, its foot must touch the ground (z = ground_z).
# Too high (not touching) and too low (through the floor) are both penalised.
def ground_cost(genome):
    z = foot_positions(genome)[..., 2]
    error = (z[STANCE] - config.ground_z) * DIST_SCALE
    return np.sum(error ** 2)


# ---------------------------------------------------------------------------
# Fitness function 4: feet lifted during swing
# ---------------------------------------------------------------------------
# While a leg is swinging forward, its foot must be lifted between
# lift_height (0.2) and max_lift (0.5) above the ground:
#   - lower than lift_height and the foot would drag along the floor
#   - higher than max_lift and the leg is waving in the air, wasting energy
# Anywhere inside that band costs nothing.
# This is what stops the robot just standing still with all feet down.
def lift_cost(genome):
    z = foot_positions(genome)[..., 2][SWING]
    too_low = np.clip(config.ground_z + config.lift_height - z, 0, None)
    too_high = np.clip(z - (config.ground_z + config.max_lift), 0, None)
    return np.sum(((too_low + too_high) * DIST_SCALE) ** 2)


# ---------------------------------------------------------------------------
# Fitness function 5: stride length
# ---------------------------------------------------------------------------
# During stance a foot is planted, so as it moves backwards relative to the
# body, the body moves forwards. The stride of a leg is how far its foot
# travels backwards (along -x) from landing to lifting off:
#   stride = x at first stance frame - x at last stance frame
# Each leg must reach at least config.stride; a longer stride costs nothing.
# A leg that steps on the spot (or backwards) gets a big penalty.
def stride_length(genome):
    x = foot_positions(genome)[..., 0]
    legs = np.arange(config.n_legs)
    return x[STANCE_START, legs] - x[STANCE_END, legs]


def stride_cost(genome):
    too_short = np.clip(config.stride - stride_length(genome), 0, None)
    return np.sum((too_short * DIST_SCALE) ** 2)


# ---------------------------------------------------------------------------
# Fitness function 6: no foot slipping
# ---------------------------------------------------------------------------
# The body is rigid, so when it moves forward every planted foot must move
# backwards by exactly the same amount (relative to the body), and the body
# should walk straight, so no foot should move sideways.
# For each step from frame t to t+1, take every leg that is on the ground in
# both frames. Their average backward movement is how far the body moved.
# Any foot moving more, less or sideways compared to that is slipping:
#   slip = foot movement - (body movement, 0)
# Without this, legs could take different stride lengths and drag their feet.
def slip_cost(genome):
    xy = foot_positions(genome)[..., :2]
    total = 0.0
    for t in range(N_FRAMES):
        nxt = (t + 1) % N_FRAMES
        planted = STANCE[t] & STANCE[nxt]
        move = xy[nxt, planted] - xy[t, planted]          # (planted legs, 2)
        body = np.array([move[:, 0].mean(), 0.0])          # forward only
        total += np.sum(((move - body) * DIST_SCALE) ** 2)
    return total


# ---------------------------------------------------------------------------
# Overall fitness
# ---------------------------------------------------------------------------
# Each cost is multiplied by its weight from config.py and added together.
# 1 / (1 + cost) turns the total into a score between 0 and 1:
#   1.0  -> a perfect walking gait: inside every cap, symmetric, stance feet
#           on the ground, swing feet lifted, every leg taking a full stride
#           and no feet slipping
#   ~0   -> large penalties
# A robot standing still now scores badly, because its feet never lift
# (lift_cost) and never push back (stride_cost).
def total_cost(genome):
    return (config.w_limit * limit_cost(genome)
            + config.w_sym * sym_cost(genome)
            + config.w_ground * ground_cost(genome)
            + config.w_lift * lift_cost(genome)
            + config.w_stride * stride_cost(genome)
            + config.w_slip * slip_cost(genome))


def fitness(genome):
    return 1.0 / (1.0 + total_cost(genome))
