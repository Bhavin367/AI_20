import csv
import math
import os
from controller import Robot

# =========================================================
# SETTINGS
# =========================================================
GAIT_FILE = "gait.csv"      # or "predicted_gait.csv" for the neural network's output
CYCLE_SECONDS = 0.8         # time for one full gait cycle (bigger = slower)
STRIDE_SCALE = 0.35         # size of each step: Webots metres per GA unit
STEP_HEIGHT = 0.15          # how high a swinging foot is lifted (m)
BODY_HEIGHT = 0.60          # how far the feet are below the hips (m)
FOOT_REACH = 0.95           # sideways distance from each hip to its foot (m)
SPLAY_DEG = 45.0            # front legs point forwards, back legs backwards
STAND_SECONDS = 3.0         # time to settle in the starting pose
PRINT_EVERY_SECONDS = 1.0

# =========================================================
# GA MODEL (must match config.py)
# =========================================================
GA_SEGMENTS = (1.2, 0.7, 1.0)                 # coxa, femur, tibia
GA_COXA_TILT = math.radians(30)
GA_BASE_DEG = [45, 75, 105, 135, -135, -105, -75, -45]    # same order as LEGS
GA_KEY_FRAMES = 4

# =========================================================
# WEBOTS ROBOT (from SpiderOctopod.proto)
# =========================================================
LEGS = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]      # CSV column order
JOINTS = ["hip_joint", "thigh_joint", "knee_joint"]
JOINT_ORDER = [f"{leg}_{joint}" for leg in LEGS for joint in JOINTS]

HIP_X = {"1": 0.4875, "2": 0.1625, "3": -0.1625, "4": -0.4875}
HIP_Y = 0.2
COXA, THIGH, TIBIA = 0.5, 0.6, 0.7
# How far each leg points forwards (+) or backwards (-), as a share of SPLAY_DEG
FORWARD_SHARE = {"1": 1.0, "2": 1 / 3, "3": -1 / 3, "4": -1.0}


# =========================================================
# KINEMATICS
# =========================================================
def rot_x(v, a):
    x, y, z = v
    return (x, y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a))


def rot_axis(v, axis, a):
    """Rotate vector v by angle a about a unit axis (Rodrigues' formula)."""
    c, s = math.cos(a), math.sin(a)
    dot = sum(i * j for i, j in zip(axis, v))
    cross = (axis[1] * v[2] - axis[2] * v[1],
             axis[2] * v[0] - axis[0] * v[2],
             axis[0] * v[1] - axis[1] * v[0])
    return tuple(v[i] * c + cross[i] * s + axis[i] * dot * (1 - c) for i in range(3))


def webots_foot(leg, q):
    """Foot position in the body frame for Webots joint angles q (radians)."""
    side = 1 if leg[0] == "L" else -1
    knee = rot_x((0.0, 0.0, -TIBIA), q[2])
    thigh = rot_x((knee[0], side * THIGH + knee[1], knee[2]), q[1])
    chain = (thigh[0], side * COXA + thigh[1], thigh[2])
    tilted = rot_x(chain, side * 0.7854)                      # coxa mounted at 45 deg
    axis = (0.0, -side * 0.707108, 0.707105)                  # tilted hip axis
    p = rot_axis(tilted, axis, q[0])
    return (HIP_X[leg[1]] + p[0], side * HIP_Y + p[1], p[2])


def solve3(A, b):
    """Solve a 3x3 linear system by Cramer's rule."""
    def det(M):
        return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    d = det(A)
    return [det([[b[r] if c == k else A[r][c] for c in range(3)] for r in range(3)]) / d
            for k in range(3)]


def inverse_kinematics(leg, target, guess):
    """Joint angles that put the foot at target (damped least squares)."""
    q = list(guess)
    for _ in range(60):
        p = webots_foot(leg, q)
        err = [t - c for t, c in zip(target, p)]
        if max(abs(e) for e in err) < 1e-5:
            break
        J = [[0.0] * 3 for _ in range(3)]
        for j in range(3):
            q2 = list(q)
            q2[j] += 1e-6
            p2 = webots_foot(leg, q2)
            for i in range(3):
                J[i][j] = (p2[i] - p[i]) / 1e-6
        JtJ = [[sum(J[k][r] * J[k][c] for k in range(3)) + (1e-4 if r == c else 0)
                for c in range(3)] for r in range(3)]
        Jte = [sum(J[k][r] * err[k] for k in range(3)) for r in range(3)]
        step = solve3(JtJ, Jte)
        q = [a + max(-0.3, min(0.3, d)) for a, d in zip(q, step)]
    return q


def ga_foot(base_deg, coxa, femur, tibia):
    """Foot position from the GA's model (same formulae as fitness.py),
    measured from where the leg joins the body."""
    heading = math.radians(base_deg + coxa)
    tilt_femur = GA_COXA_TILT + math.radians(femur)
    tilt_tibia = tilt_femur + math.radians(tibia)
    lengths_tilts = list(zip(GA_SEGMENTS, [GA_COXA_TILT, tilt_femur, tilt_tibia]))
    reach = sum(s * math.cos(t) for s, t in lengths_tilts)
    z = sum(s * math.sin(t) for s, t in lengths_tilts)
    return (reach * math.cos(heading), reach * math.sin(heading), z)


# =========================================================
# LOAD THE GAIT AND PLAN THE FOOT PATHS
# =========================================================
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), GAIT_FILE)
if not os.path.exists(csv_path):
    raise SystemExit(f"Could not find {csv_path}")
with open(csv_path, newline="") as f:
    rows = [[float(v) for v in row] for row in csv.reader(f) if row]
if (len(rows) < GA_KEY_FRAMES or len(rows) % GA_KEY_FRAMES
        or any(len(r) != len(JOINT_ORDER) for r in rows)):
    raise SystemExit(f"{GAIT_FILE} needs a multiple of {GA_KEY_FRAMES} rows, "
                     f"each with {len(JOINT_ORDER)} values")
keys = rows[::len(rows) // GA_KEY_FRAMES]
print(f"Loaded {len(rows)} rows from {csv_path}; using {len(keys)} key frames")


def foot_path(i, leg):
    """The Webots foot position for one leg in each key frame."""
    side = 1 if leg[0] == "L" else -1
    ga = [ga_foot(GA_BASE_DEG[i], *k[3 * i:3 * i + 3]) for k in keys]
    ground = sorted(p[2] for p in ga)[:3]                  # the 3 stance frames
    ground_z = sum(ground) / 3
    swing = max(range(len(ga)), key=lambda t: ga[t][2])    # highest = swing frame
    stance = [p for t, p in enumerate(ga) if t != swing]
    mid_x = (max(p[0] for p in stance) + min(p[0] for p in stance)) / 2
    mid_y = sum(p[1] for p in stance) / len(stance)

    # Where the foot rests: out to the side, angled forwards or backwards
    angle = math.radians(SPLAY_DEG * FORWARD_SHARE[leg[1]])
    rest_x = HIP_X[leg[1]] + FOOT_REACH * math.sin(angle)
    rest_y = side * (HIP_Y + FOOT_REACH * math.cos(angle))

    path = []
    for t, (x, y, z) in enumerate(ga):
        if t == swing:
            # swing foot: lifted, halfway between lift-off and landing
            path.append((rest_x, rest_y, -BODY_HEIGHT + STEP_HEIGHT))
        else:
            path.append((rest_x + STRIDE_SCALE * (x - mid_x),
                         rest_y + STRIDE_SCALE * (y - mid_y),
                         -BODY_HEIGHT))
    return path


# Solve the joint angles for many points round the cycle before walking
SAMPLES = 200
table = [[0.0] * len(JOINT_ORDER) for _ in range(SAMPLES)]
worst = 0.0
for i, leg in enumerate(LEGS):
    path = foot_path(i, leg)
    side = 1 if leg[0] == "L" else -1
    q = [0.0, -side * 1.0, side * 0.5]                     # a bent-leg first guess
    for pass_number in range(2):                           # 2nd pass closes the loop
        for s in range(SAMPLES):
            pos = s / SAMPLES * len(path)
            k, frac = int(pos), pos - int(pos)
            a, b = path[k], path[(k + 1) % len(path)]
            target = [u + frac * (v - u) for u, v in zip(a, b)]
            q = inverse_kinematics(leg, target, q)
            if pass_number == 1:
                table[s][3 * i:3 * i + 3] = q
                worst = max(worst, math.dist(target, webots_foot(leg, q)))
if worst > 0.01:
    print(f"WARNING: some foot positions are out of reach (by up to {worst:.3f} m)")

# =========================================================
# MOTORS AND GPS
# =========================================================
robot = Robot()
timestep = int(robot.getBasicTimeStep())

gps = robot.getDevice("gps")
gps.enable(timestep)

motors, limits = [], []
for name in JOINT_ORDER:
    motor = robot.getDevice(name)
    if motor is None:
        raise SystemExit(f"Motor '{name}' not found")
    motor.setVelocity(motor.getMaxVelocity())
    motors.append(motor)
    limits.append((motor.getMinPosition(), motor.getMaxPosition()))

clamped = 0


def command(angles):
    """Send 24 angles to the motors, keeping each inside its limits."""
    global clamped
    for motor, (lo, hi), angle in zip(motors, limits, angles):
        safe = min(max(angle, lo), hi)
        if abs(safe - angle) > 1e-6:
            clamped += 1
        motor.setPosition(safe)


def pose_at(phase):
    """Joint angles at a point in the cycle (0 to 1)."""
    pos = (phase % 1.0) * SAMPLES
    k = int(pos) % SAMPLES
    frac = pos - int(pos)
    a, b = table[k], table[(k + 1) % SAMPLES]
    return [x + frac * (y - x) for x, y in zip(a, b)]


# A quarter of the way through the cycle every foot is on the ground
# (GA key frame 1), so it is a stable pose to start from.
START_PHASE = 0.25

# =========================================================
# STAND, THEN WALK
# =========================================================
command(pose_at(START_PHASE))
stand_start = robot.getTime()
while robot.getTime() - stand_start < STAND_SECONDS:
    if robot.step(timestep) == -1:
        raise SystemExit

start_xyz = gps.getValues()
print(f"Standing, body height Z = {start_xyz[2]:.3f}. Starting to walk.")

walk_start = robot.getTime()
next_print = PRINT_EVERY_SECONDS
while robot.step(timestep) != -1:
    t = robot.getTime() - walk_start
    command(pose_at(START_PHASE + t / CYCLE_SECONDS))
    if t >= next_print:
        next_print += PRINT_EVERY_SECONDS
        x, y, z = gps.getValues()
        travelled = math.hypot(x - start_xyz[0], y - start_xyz[1])
        print(f"t={t:5.1f}s  X={x:.3f} Y={y:.3f} Z={z:.3f}  "
              f"distance={travelled:.3f} m  clamped={clamped}")
