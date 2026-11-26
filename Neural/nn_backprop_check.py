"""
nn_backprop_check.py - evidence that the network is correct and works.

    1. Gradient check : backprop against a numerical gradient
    2. Evaluation     : errors in degrees, plus a joint limit check
    3. Plots          : convergence, predicted vs target, error per joint, poses

Nothing runs on import. Use main.py.
"""
import matplotlib.pyplot as plt
import numpy as np

import neuralnetwork as nn

BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"


# ===========================================================================
# 1. GRADIENT CHECK
# ===========================================================================
# A falling loss does not prove backprop is right: a wrong gradient can still
# reduce the loss. So each gradient is checked a second, independent way. For
# a weight w, nudge it up and down by a small amount e:
#
#   numerical gradient = (Loss(w + e) - Loss(w - e)) / (2e)
#
# and compare it with backprop using the relative error
#
#   |numerical - backprop| / (|numerical| + |backprop|)
def gradient_check(X, Y, layers, activation="tanh", n_samples=20,
                   checks_per_param=25, eps=1e-6, seed=0):
    """Check randomly chosen weights and biases. Returns the worst error."""
    rng = np.random.default_rng(seed)
    params = nn.init_weights(layers, rng, activation)
    idx_samples = rng.choice(len(X), size=n_samples, replace=False)
    X, Y = X[idx_samples], Y[idx_samples]

    def loss():
        return nn.mse_loss(nn.predict(X, params, activation), Y)

    pred, acts = nn.forward_with_cache(X, params, activation)
    grads = nn.backward(pred, Y, acts, params, activation)

    print(f"  {'parameter':<14}{'shape':<12}{'worst relative error':>22}")
    worst_overall = 0.0
    for i, (layer, layer_grads) in enumerate(zip(params, grads)):
        for name, p, g in zip(["W", "b"], layer, layer_grads):
            worst = 0.0
            for _ in range(checks_per_param):
                idx = tuple(rng.integers(s) for s in p.shape)
                original = p[idx]
                p[idx] = original + eps
                loss_up = loss()
                p[idx] = original - eps
                loss_down = loss()
                p[idx] = original                      # put the weight back
                numerical = (loss_up - loss_down) / (2 * eps)
                # The 1e-7 floor stops rounding noise on near-zero gradients
                # from looking like a large relative error
                scale = max(abs(numerical) + abs(g[idx]), 1e-7)
                worst = max(worst, abs(numerical - g[idx]) / scale)
            worst_overall = max(worst_overall, worst)
            print(f"  layer {i + 1} {name:<6}{str(p.shape):<12}{worst:>22.2e}")
    return worst_overall


# ===========================================================================
# 2. EVALUATION
# ===========================================================================
def error_summary(pred, Y):
    """Loss in radians squared, and angle errors in degrees."""
    err = np.abs(np.rad2deg(pred - Y))
    return {"mse": nn.mse_loss(pred, Y), "mean_deg": err.mean(), "max_deg": err.max()}


def evaluate(params, activation, keyframes_deg):
    """Test on three sets of inputs and return printable lines.

    - training samples
    - in-between phases the network never saw (halfway between samples)
    - the GA's own key frames

    Limitation: the in-between targets come from the same interpolation as the
    training data, so this shows the network is smooth between samples. It does
    not show it could invent a different gait.
    """
    n_frames = len(keyframes_deg)
    X_train, Y_train, _ = nn.make_dataset(keyframes_deg)
    X_between, Y_between, _ = nn.make_dataset(keyframes_deg, offset=0.5)
    X_key = nn.phase_to_input(np.arange(n_frames) / n_frames)
    Y_key = np.deg2rad(keyframes_deg)

    lines = [f"  {'test set':<24}{'MSE (rad^2)':>14}{'mean err (deg)':>17}{'max err (deg)':>16}"]
    for name, X, Y in [("training samples", X_train, Y_train),
                       ("in-between phases", X_between, Y_between),
                       (f"GA key frames ({n_frames})", X_key, Y_key)]:
        s = error_summary(nn.predict(X, params, activation), Y)
        lines.append(f"  {name:<24}{s['mse']:>14.7f}{s['mean_deg']:>17.3f}{s['max_deg']:>16.3f}")
    lines.append("  " + joint_limit_report(nn.predict(X_between, params, activation)))
    return lines


def joint_limit_report(pred):
    """Are the predicted angles valid, i.e. inside the joint caps in config.py?"""
    try:
        import config
        low = np.asarray(config.low, dtype=float).ravel()
        high = np.asarray(config.high, dtype=float).ravel()
    except (ImportError, AttributeError):
        return "Joint limit check skipped (config.py with low/high not found)."
    if low.size != nn.N_JOINTS or high.size != nn.N_JOINTS:
        return "Joint limit check skipped (config.low/high do not have 24 values)."
    excess = np.rad2deg(np.maximum(low - pred, pred - high))    # > 0 means outside
    outside = excess > 1e-6
    if not outside.any():
        return "Joint limit check: all predicted angles are within the limits."
    return (f"Joint limit check: {outside.mean():.2%} of predicted angles are outside "
            f"the limits (worst by {excess.max():.3f} deg).")


# ===========================================================================
# 3. PLOTS
# ===========================================================================
def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(history, title, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history, color=BLUE, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    _finish(fig, path)


def plot_predicted_vs_target(pred, Y, phase, n_frames, path):
    """All 24 joints: one panel per leg. Thick faint = target, dashed = network.
    Dotted vertical lines mark the GA key frames."""
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True, sharey=True)
    # Left legs in the left column, right legs in the right column, 1 to 4 down
    order = [nn.LEG_NAMES.index(f"{side}{n}") for n in range(1, 5) for side in "LR"]
    for ax, leg in zip(axes.flat, order):
        for k, (joint, colour) in enumerate(zip(nn.JOINT_TYPES, [BLUE, ORANGE, AQUA])):
            j = leg * 3 + k
            ax.plot(phase, np.rad2deg(Y[:, j]), color=colour, linewidth=4, alpha=0.35)
            ax.plot(phase, np.rad2deg(pred[:, j]), "--", color=colour,
                    linewidth=1.5, label=joint)
        for frame in range(n_frames if n_frames <= 12 else 0):
            ax.axvline(frame / n_frames, color="grey", linestyle=":", linewidth=0.8)
        ax.set_title(nn.LEG_NAMES[leg])
        ax.grid(True, alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel("Angle (degrees)")
    for ax in axes[-1]:
        ax.set_xlabel("Phase of walk cycle")
    fig.suptitle("Target (thick) vs network (dashed), at in-between phases")
    axes[0, 0].legend(fontsize=8)
    _finish(fig, path)


def plot_joint_errors(pred, Y, path):
    """Mean absolute error for each of the 24 joints."""
    err = np.abs(np.rad2deg(pred - Y)).mean(axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(np.arange(nn.N_JOINTS), err, color=BLUE, width=0.75)
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(nn.N_JOINTS), nn.JOINT_NAMES, rotation=90, fontsize=8)
    ax.set_ylabel("Mean absolute error (degrees)")
    ax.set_title("Error per joint, at in-between phases")
    ax.grid(True, axis="y", alpha=0.25)
    _finish(fig, path)


# ---- Spider poses ---------------------------------------------------------
# A simplified stick model for drawing only. CHECK THESE against the model the
# GA uses (lengths, leg directions and the sign of each angle) and edit them to
# match. They change the picture, not the training or the error numbers.
BODY_RADIUS = 0.5
SEGMENTS = (0.3, 0.8, 1.0)                           # coxa, femur, tibia lengths
LEG_DIRECTION = {"L1": 45, "L2": 75, "L3": 105, "L4": 135,       # degrees from the
                 "R1": -45, "R2": -75, "R3": -105, "R4": -135}   # forward (x) axis


def leg_points(angles, leg_name):
    """3D positions of one leg's 4 points from its (coxa, femur, tibia) angles.
    coxa swings the leg horizontally, femur raises it, tibia bends relative
    to the femur."""
    coxa, femur, tibia = angles
    base = np.deg2rad(LEG_DIRECTION[leg_name])
    side = 1 if leg_name.startswith("L") else -1
    heading = base + side * coxa
    hip = BODY_RADIUS * np.array([np.cos(base), np.sin(base), 0.0])
    points = [hip]
    for length, elevation in zip(SEGMENTS, [0.0, femur, femur + tibia]):
        step = length * np.array([np.cos(elevation) * np.cos(heading),
                                  np.cos(elevation) * np.sin(heading),
                                  np.sin(elevation)])
        points.append(points[-1] + step)
    return np.array(points)


def plot_poses(params, activation, keyframes_deg, path, max_poses=4):
    """Target and predicted spider drawn on top of each other at key frames."""
    n_frames = len(keyframes_deg)
    chosen = np.unique(np.linspace(0, n_frames, min(max_poses, n_frames),
                                   endpoint=False).astype(int))
    target = np.deg2rad(keyframes_deg[chosen])
    pred = nn.predict(nn.phase_to_input(chosen / n_frames), params, activation)

    fig = plt.figure(figsize=(5 * len(chosen), 5.5))
    for i, frame in enumerate(chosen):
        ax = fig.add_subplot(1, len(chosen), i + 1, projection="3d")
        for angles, colour, style, width, label in [
                (target[i], BLUE, "-", 4, "Target (GA)"),
                (pred[i], ORANGE, "--", 1.8, "Network")]:
            for leg, name in enumerate(nn.LEG_NAMES):
                p = leg_points(angles[leg * 3:leg * 3 + 3], name)
                ax.plot(p[:, 0], p[:, 1], p[:, 2], style, color=colour, linewidth=width,
                        alpha=0.5 if style == "-" else 1.0,
                        label=label if leg == 0 else None)
                if style == "-":
                    ax.text(*p[-1], name, fontsize=7, color="#333333")
        around = np.linspace(0, 2 * np.pi, 50)
        ax.plot(BODY_RADIUS * np.cos(around), BODY_RADIUS * np.sin(around),
                np.zeros(50), color="grey")
        reach = BODY_RADIUS + sum(SEGMENTS)
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_zlim(-reach, reach)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=28, azim=-60)
        error = np.abs(np.rad2deg(pred[i] - target[i])).mean()
        ax.set_title(f"Key frame {frame}\nmean error {error:.2f} deg")
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
    # tight_layout does not handle 3D axes well, so set the margins directly
    fig.subplots_adjust(left=0.01, right=0.97, bottom=0.03, top=0.88, wspace=0.08)
    fig.savefig(path, dpi=150)
    plt.close(fig)
