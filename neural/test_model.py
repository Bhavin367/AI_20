import argparse
import os
import numpy as np

NEURAL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(NEURAL_DIR)
MODEL_PATH = os.path.join(NEURAL_DIR, "trained_model.npz")
LEG_NAMES = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]


def load_model(path=MODEL_PATH):
    """Load saved MLP weights, layer sizes, and cycle length from .npz."""
    data = np.load(path)
    layer_sizes = data["layer_sizes"].tolist()
    n_frames = int(data["n_frames"])
    params = [(data[f"W{i}"], data[f"b{i}"]) for i in range(len(layer_sizes) - 1)]
    return params, layer_sizes, n_frames


def forward(X_in, params):
    """Run MLP forward pass (tanh hidden layers, linear output)."""
    a = X_in
    for W, b in params[:-1]:
        a = np.tanh(a @ W + b)
    W, b = params[-1]
    return a @ W + b


def predict_phase(phi, params):
    """Predict 24 joint angles (radians) for arbitrary phase angle phi (radians)."""
    phi_arr = np.atleast_1d(np.asarray(phi, dtype=float))
    X = np.column_stack([np.sin(phi_arr), np.cos(phi_arr)])
    pred = forward(X, params)
    return pred[0] if np.ndim(phi) == 0 else pred


def predict_frame(frame, params, n_frames=300):
    """Predict 24 joint angles (radians) for frame index in 0..n_frames-1 (supports floats)."""
    return predict_phase(2.0 * np.pi * np.asarray(frame, dtype=float) / n_frames, params)


def print_pose(angles_rad, label):
    """Pretty-print 24 joint angles grouped by leg in degrees and radians."""
    angles_deg = np.rad2deg(angles_rad).reshape(8, 3)
    angles_r = angles_rad.reshape(8, 3)
    print(f"\n--- {label} ---")
    print(f"{'Leg':<4} {'Coxa (deg/rad)':>18} {'Femur (deg/rad)':>18} {'Tibia (deg/rad)':>18}")
    for name, d, r in zip(LEG_NAMES, angles_deg, angles_r):
        print(
            f"{name:<4} "
            f"{d[0]:7.2f} / {r[0]:6.3f}   "
            f"{d[1]:7.2f} / {r[1]:6.3f}   "
            f"{d[2]:7.2f} / {r[2]:6.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Test saved spider NN model.")
    parser.add_argument("--frame", type=float, default=None, help="Frame in 0..299 (float allowed)")
    parser.add_argument("--phi", type=float, default=None, help="Phase angle phi in radians")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to trained_model.npz")
    args = parser.parse_args()

    params, layer_sizes, n_frames = load_model(args.model)
    n_params = sum(W.size + b.size for W, b in params)
    print(f"Loaded {args.model} | architecture={layer_sizes} ({n_params} params) | n_frames={n_frames}")

    if args.frame is not None:
        pred = predict_frame(args.frame, params, n_frames)
        print_pose(pred, f"Predicted Pose at frame={args.frame:g}")
        return

    if args.phi is not None:
        pred = predict_phase(args.phi, params)
        print_pose(pred, f"Predicted Pose at phi={args.phi:.4f} rad")
        return

    all_frames = np.arange(n_frames)
    all_pred_rad = predict_frame(all_frames, params, n_frames)

    pred_csv = os.path.join(NEURAL_DIR, "predicted_gait_rad.csv")
    if os.path.exists(pred_csv):
        saved_pred = np.loadtxt(pred_csv, delimiter=",")
        max_diff = np.max(np.abs(all_pred_rad - saved_pred))
        print(f"Exact match check vs predicted_gait_rad.csv: max abs diff = {max_diff:.2e} rad")

    target_csv = os.path.join(BASE_DIR, "gait_rad.csv")
    if os.path.exists(target_csv):
        target_rad = np.loadtxt(target_csv, delimiter=",")
        rmse_deg = np.degrees(np.sqrt(np.mean((all_pred_rad - target_rad) ** 2)))
        print(f"Full-cycle RMSE vs GA target (gait_rad.csv): {rmse_deg:.3f} deg")

    for f in [0, 75, 150, 225]:
        print_pose(all_pred_rad[f], f"Key Phase Frame {f}/{n_frames}")

    unseen_phi = np.pi / 4.0
    print_pose(predict_phase(unseen_phi, params), f"Unseen Intermediate Phase phi=pi/4 ({unseen_phi:.4f} rad)")


if __name__ == "__main__":
    main()
