import csv
import importlib
import math
import os

try:
    Robot = importlib.import_module("controller").Robot
    IN_WEBOTS = True
except ModuleNotFoundError:
    Robot = None
    IN_WEBOTS = False

# =========================================================
# SETTINGS (edit these)
# =========================================================

GAIT_FILE = "predicted_gait.csv"         # use "predicted_gait.csv" to play the neural network's output
GAIT_IN_DEGREES = True          # False if the CSV is already in radians
GAIT_HAS_HEADER = False         # True if the first row of the CSV is a header
MIN_FRAMES = 100                # fewer rows than this = keyframes, so interpolate up to TARGET_FRAMES
TARGET_FRAMES = 300
HIP_SIGN = 1
INVERT_RIGHT_THIGH_KNEE = True  # flip sign on right thigh/knee (R thighs are + when standing, L are -)
THIGH_OFFSET = math.radians(-5.0)
KNEE_OFFSET = math.radians(62.0)

STAND_VELOCITY = 1.5
WALK_VELOCITY = 2.0
LOWER_STEP = math.radians(0.5)
TARGET_Z = 0.300
MAX_LOWER_STEPS = 2000          # safety cap so the height loop can never hang
BLEND_SECONDS = 1.0             # ease from standing pose into frame 0 of the gait
PRINT_EVERY_SECONDS = 1.0       # how often to print GPS position while walking

# =========================================================
# ROBOT, LEGS AND JOINT ORDER
# =========================================================

if IN_WEBOTS:
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

right_legs = ["R1", "R2", "R3", "R4"]
left_legs = ["L1", "L2", "L3", "L4"]
all_legs = right_legs + left_legs

# Column order of gait.csv
GAIT_LEGS = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
JOINT_ORDER = [
    f"{leg}_{joint}"
    for leg in GAIT_LEGS
    for joint in ["hip_joint", "thigh_joint", "knee_joint"]
]

TARGET_ANGLES = {
    "R": {"hip_joint": 0.0, "thigh_joint": 1.22, "knee_joint": -0.30},
    "L": {"hip_joint": 0.0, "thigh_joint": -1.22, "knee_joint": 0.30},
}

# =========================================================
# GPS AND MOTORS
# =========================================================

if IN_WEBOTS:
    gps = robot.getDevice("gps")
    if gps is None:
        print("ERROR: GPS not found!")
        exit()
    gps.enable(timestep)

    motors = {}
    for name in JOINT_ORDER:
        motor = robot.getDevice(name)
        if motor:
            motors[name] = motor
            motor.setVelocity(STAND_VELOCITY)
        else:
            print(f"WARNING: Motor '{name}' not found!")

    # Last position commanded to each motor (used to blend smoothly)
    commanded = {name: 0.0 for name in motors}


def set_pos(name, angle_rad):
    """Command a motor and remember what we asked for."""
    if name in motors:
        motors[name].setPosition(angle_rad)
        commanded[name] = angle_rad


def step_or_exit():
    if robot.step(timestep) == -1:
        exit()


def joint_target(name, angle_rad):
    """Apply the right-side sign flip used by the gait."""
    if "hip" in name:
        angle_rad *= HIP_SIGN
    elif "thigh" in name:
        angle_rad += THIGH_OFFSET
    elif "knee" in name:
        angle_rad += KNEE_OFFSET

    if INVERT_RIGHT_THIGH_KNEE and name.startswith("R"):
        if "thigh" in name or "knee" in name:
            angle_rad = -angle_rad

    if "knee" in name:
        return max(-2.0, min(1.0, angle_rad))
    return max(-1.5708, min(1.5708, angle_rad))


# =========================================================
# LOAD GAIT CSV (same folder as this file)
# =========================================================

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), GAIT_FILE)

if not os.path.exists(csv_path):
    print(f"ERROR: Could not find {GAIT_FILE} at {csv_path}")
    exit()

with open(csv_path, newline="") as f:
    reader = csv.reader(f)
    if GAIT_HAS_HEADER:
        next(reader, None)
    gait_rows = [[float(v) for v in row] for row in reader if row]

if not gait_rows or any(len(r) != len(JOINT_ORDER) for r in gait_rows):
    print(f"ERROR: every row of {GAIT_FILE} must have {len(JOINT_ORDER)} values")
    exit()

if GAIT_IN_DEGREES:
    gait_rows = [[math.radians(v) for v in row] for row in gait_rows]

print(f"Loaded {len(gait_rows)} gait rows from {csv_path}")

# Few rows means keyframes: interpolate to a smooth looping cycle
if len(gait_rows) < MIN_FRAMES:
    n_key = len(gait_rows)
    looped = gait_rows + [gait_rows[0]]          # frame 0 again at the end so the cycle loops
    smooth = []
    for i in range(TARGET_FRAMES):
        pos = i / TARGET_FRAMES * n_key          # position along the keyframes
        k = int(pos)
        frac = pos - k
        a, b = looped[k], looped[k + 1]
        smooth.append([a[j] + frac * (b[j] - a[j]) for j in range(len(JOINT_ORDER))])
    gait_rows = smooth
    print(f"Interpolated {n_key} keyframes up to {len(gait_rows)} frames")

num_frames = len(gait_rows)

if not IN_WEBOTS:
    hip_vals, thigh_vals, knee_vals = [], [], []
    for row in gait_rows:
        for name, angle in zip(JOINT_ORDER, row):
            cmd = joint_target(name, angle)
            if not math.isfinite(cmd):
                raise ValueError(f"Non-finite joint target for {name}: {cmd}")
            if "hip" in name:
                hip_vals.append(cmd)
            elif "thigh" in name:
                thigh_vals.append(cmd)
            else:
                knee_vals.append(cmd)
    print("[Offline check] 'controller' module not active (running outside Webots).")
    print(f"Verified {num_frames} frames x {len(JOINT_ORDER)} joints for {GAIT_FILE}:")
    print(f"  hip   range (rad): [{min(hip_vals):.3f}, {max(hip_vals):.3f}] (limit [-1.571, 1.571])")
    print(f"  thigh range (rad): [{min(thigh_vals):.3f}, {max(thigh_vals):.3f}] (limit [-1.571, 1.571])")
    print(f"  knee  range (rad): [{min(knee_vals):.3f}, {max(knee_vals):.3f}] (limit [-2.000, 1.000])")
    print("Open Spider_world_demo/worlds/spider_world_1.wbt in Webots and press Play to simulate in 3D.")
    raise SystemExit(0)

# =========================================================
# STAND UP
# =========================================================

print("\n==============================================")
print("SETTING INITIAL STANDING POSITION")
print("==============================================")

for leg in right_legs + left_legs:
    side = "R" if leg.startswith("R") else "L"
    for joint in ["hip_joint", "thigh_joint", "knee_joint"]:
        set_pos(f"{leg}_{joint}", TARGET_ANGLES[side][joint])

for _ in range(int(3.0 * 1000 / timestep)):
    step_or_exit()

# =========================================================
# ADJUST HEIGHT TO TARGET Z
# =========================================================

position = gps.getValues()
print(f"\nInitial GPS: X={position[0]:.4f} Y={position[1]:.4f} Z={position[2]:.4f}")
print(f"Target Z = {TARGET_Z:.4f} m")

right_thigh = TARGET_ANGLES["R"]["thigh_joint"]
right_knee = TARGET_ANGLES["R"]["knee_joint"]
left_thigh = TARGET_ANGLES["L"]["thigh_joint"]
left_knee = TARGET_ANGLES["L"]["knee_joint"]

print("\n==============================================")
print("ADJUSTING ROBOT HEIGHT")
print("==============================================")

for i in range(MAX_LOWER_STEPS):
    if robot.step(timestep) == -1:
        exit()
    current_z = gps.getValues()[2]

    if current_z >= TARGET_Z:
        print(f"Target height reached: Z = {current_z:.4f} m")
        break

    right_thigh += LOWER_STEP
    right_knee -= LOWER_STEP
    left_thigh -= LOWER_STEP
    left_knee += LOWER_STEP

    for leg in right_legs:
        set_pos(f"{leg}_thigh_joint", right_thigh)
        set_pos(f"{leg}_knee_joint", right_knee)
    for leg in left_legs:
        set_pos(f"{leg}_thigh_joint", left_thigh)
        set_pos(f"{leg}_knee_joint", left_knee)

    if i % 20 == 0:
        print(f"Z = {current_z:.4f} m | R thigh = {math.degrees(right_thigh):.1f} deg | "
              f"L thigh = {math.degrees(left_thigh):.1f} deg")
else:
    print(f"WARNING: height loop hit the {MAX_LOWER_STEPS}-step cap, Z = {gps.getValues()[2]:.4f} m")

# =========================================================
# BLEND FROM STANDING POSE INTO FRAME 0
# =========================================================

print("\n==============================================")
print("BLENDING INTO GAIT")
print("==============================================")

for name in motors:
    motors[name].setVelocity(WALK_VELOCITY)

start = dict(commanded)
first = {name: joint_target(name, angle)
         for name, angle in zip(JOINT_ORDER, gait_rows[0])}
blend_steps = max(1, int(BLEND_SECONDS * 1000 / timestep))

for i in range(1, blend_steps + 1):
    a = i / blend_steps
    for name in motors:
        set_pos(name, (1 - a) * start[name] + a * first[name])
    step_or_exit()

# =========================================================
# WALKING LOOP: PLAY THE GAIT ON REPEAT
# =========================================================

print("\n==============================================")
print("STARTING WALKING GAIT")
print("==============================================")

start_xyz = gps.getValues()
print_every = max(1, int(PRINT_EVERY_SECONDS * 1000 / timestep))
frame = 0

while robot.step(timestep) != -1:
    row = gait_rows[frame % num_frames]

    for name, angle in zip(JOINT_ORDER, row):
        set_pos(name, joint_target(name, angle))

    frame += 1

    if frame % print_every == 0:
        x, y, z = gps.getValues()
        travelled = math.hypot(x - start_xyz[0], y - start_xyz[1])
        print(f"cycle {frame // num_frames} | X={x:.3f} Y={y:.3f} Z={z:.3f} | "
              f"distance from start = {travelled:.3f} m")
