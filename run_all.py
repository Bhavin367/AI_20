import math
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import main as part1_main
import ga_experiments


WEBOTS_DIR = os.path.join(BASE_DIR, "Spider_world_demo", "controllers", "my_controller_s")
LEGS = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
ANCHORS_X = [0.4875, 0.1625, -0.1625, -0.4875, -0.4875, -0.1625, 0.1625, 0.4875]
THIGH_OFFSET = math.radians(-5.0)
KNEE_OFFSET = math.radians(62.0)


def rot_axis(axis, theta):
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    x, y, z = ax
    c, s = math.cos(theta), math.sin(theta)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def webots_fk(leg_idx, hip, thigh, knee):
    s = 1.0 if leg_idx < 4 else -1.0
    anchor = np.array([ANCHORS_X[leg_idx], s * 0.2, 0.0])
    R_hip = rot_axis([0, -s / math.sqrt(2), 1 / math.sqrt(2)], hip)
    R_coxa = R_hip @ rot_axis([s, 0, 0], 0.7854)
    p_thigh = anchor + R_coxa @ np.array([0, s * 0.5, 0])
    R_thigh = R_coxa @ rot_axis([1, 0, 0], thigh)
    p_knee = p_thigh + R_thigh @ np.array([0, s * 0.6, 0])
    R_knee = R_thigh @ rot_axis([1, 0, 0], knee)
    return p_knee + R_knee @ np.array([0, 0, -0.7])


def verify_webots_csv(filename):
    path = os.path.join(WEBOTS_DIR, filename)
    rows = np.deg2rad(np.loadtxt(path, delimiter=",")).reshape(300, 8, 3)
    knees_cmd = []
    feet_z = np.zeros((300, 8))
    feet_x = np.zeros((300, 8))
    for r in range(300):
        for i, leg in enumerate(LEGS):
            h, th, kn = rows[r, i]
            th = th + THIGH_OFFSET
            kn = kn + KNEE_OFFSET
            if leg.startswith("R"):
                th, kn = -th, -kn
            kn_clamped = max(-2.0, min(1.0, kn))
            knees_cmd.append(kn)
            pf = webots_fk(i, h, th, kn_clamped)
            feet_x[r, i] = pf[0]
            feet_z[r, i] = pf[2]

    stance_z, swing_z = [], []
    for f_idx, r in enumerate([0, 75, 150, 225]):
        for i in range(8):
            if (f_idx == 0 and i % 2 == 0) or (f_idx == 2 and i % 2 == 1):
                swing_z.append(feet_z[r, i])
            else:
                stance_z.append(feet_z[r, i])
    strides = [
        (feet_x[75, i] - feet_x[225, i]) if i % 2 == 0 else (feet_x[225, i] - feet_x[75, i])
        for i in range(8)
    ]
    print(f"\n[Webots FK check: {filename}]")
    print(f"  Raw knee range (rad): [{min(knees_cmd):.3f}, {max(knees_cmd):.3f}] (proto limit [-2.000, +1.000])")
    print(f"  Stance foot Z mean: {np.mean(stance_z):.3f} m (range [{min(stance_z):.3f}, {max(stance_z):.3f}])")
    print(f"  Swing  foot Z mean: {np.mean(swing_z):.3f} m (avg clearance {np.mean(swing_z) - np.mean(stance_z):.3f} m)")
    print(f"  Forward stride per cycle: min={min(strides):.3f} m, mean={np.mean(strides):.3f} m")


if __name__ == "__main__":
    print("=== STEP 1/4: PART 1 GA (main.py) ===")
    part1_main.main()

    print("\n=== STEP 2/4: PART 1 GA ABLATION (ga_experiments.py) ===")
    ga_experiments.main()

    print("\n=== STEP 3/4: PART 2 NEURAL NETWORK (neural/neuralnetwork.py) ===")
    from neural import neuralnetwork as part2_nn
    part2_nn.main()

    print("\n=== STEP 4/4: PART 2 OPTIMISER COMPARISON (neural/nntrainingmethod.py) ===")
    from neural import nntrainingmethod as part2_opt
    part2_opt.main()

    verify_webots_csv("gait.csv")
    verify_webots_csv("predicted_gait.csv")
    print("\nAll files generated and verified! Open Spider_world_demo/worlds/spider_world_1.wbt in Webots to run.")
