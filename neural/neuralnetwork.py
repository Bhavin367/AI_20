import os
import sys
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)  # fixed seed so results repeat

NEURAL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(NEURAL_DIR)
WEBOTS_DIR = os.path.join(BASE_DIR, "Spider_world_demo", "controllers", "my_controller_s")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
import fitness as fit
import viz


def load_ga_gait_radians():
    rad_path = os.path.join(BASE_DIR, "gait_rad.csv")
    deg_path = os.path.join(BASE_DIR, "gait.csv")
    if os.path.exists(rad_path):
        Y_rad = np.loadtxt(rad_path, delimiter=",")
        source = rad_path
    elif os.path.exists(deg_path):
        Y_rad = np.deg2rad(np.loadtxt(deg_path, delimiter=","))
        source = deg_path + " (converted to radians)"
    else:
        raise FileNotFoundError(f"Could not find gait_rad.csv or gait.csv in {BASE_DIR}. Run main.py (Part 1) first.")
    print(f"Loaded target gait from {source}, shape: {Y_rad.shape}")
    return Y_rad


# ===== STEP 1: TRAINING DATA =====
# 300 frames in one walk cycle. Each frame becomes 2 numbers (sin, cos),
# so frame 299 sits right next to frame 0: the network learns the walk loops.
n_frames = 300
t = np.arange(n_frames)
phase = 2 * np.pi * t / n_frames
X = np.column_stack([np.sin(phase), np.cos(phase)])   # inputs, shape (300, 2)
Y = load_ga_gait_radians()                            # targets, shape (300, 24)

train_idx = np.arange(0, n_frames, 2)
val_idx = np.arange(1, n_frames, 2)
X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]

print("X shape:", X.shape, "| Y shape:", Y.shape)
print(f"Train frames: {len(train_idx)} (even) | Validation frames: {len(val_idx)} (odd)")


# ===== STEP 2: ARCHITECTURE / WEIGHTS =====
# 2 inputs -> 64 -> 64 -> 24 outputs
layer_sizes = [2, 64, 64, 24]


def init_weights(sizes, local_rng=None):
    """One weight matrix W and bias vector b per layer.
    W has shape (neurons in, neurons out). Starting values are small and
    random, scaled by 1/sqrt(inputs), so signals don't blow up or vanish."""
    r = rng if local_rng is None else local_rng
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        W = r.normal(0, 1 / np.sqrt(n_in), size=(n_in, n_out))
        b = np.zeros(n_out)
        params.append((W, b))
    return params


# ===== STEP 3: FORWARD PASS =====
def forward(X_in, params):
    """Push inputs through the network.
    Hidden layers: weighted sum, then tanh (smooth, outputs between -1 and 1).
    Output layer: weighted sum only (no activation), since angles can be any value."""
    a = X_in
    for W, b in params[:-1]:
        a = np.tanh(a @ W + b)       # hidden layer
    W, b = params[-1]
    return a @ W + b                 # output layer: 24 angles


# ===== STEP 4: LOSS =====
def mse_loss(pred, target):
    """Mean squared error: average of (prediction - target)^2 over every
    angle in every frame. Squaring makes all errors positive and punishes
    big mistakes more than small ones. 0 = perfect match."""
    return np.mean((pred - target) ** 2)


# ===== STEP 5: BACKPROPAGATION =====
def forward_with_cache(X_in, params):
    """Same as forward(), but also keeps each layer's output,
    because backprop needs them to compute gradients."""
    activations = [X_in]             # activations[0] = the input
    a = X_in
    for W, b in params[:-1]:
        a = np.tanh(a @ W + b)
        activations.append(a)     # output of each hidden layer
    W, b = params[-1]
    pred = a @ W + b
    return pred, activations


def backward(pred, target, activations, params):
    """Work out the gradient of the MSE loss for every weight and bias,
    moving backwards from the output layer to the first layer."""
    grads = [None] * len(params)

    # How the loss changes with each prediction (derivative of MSE)
    delta = 2 * (pred - target) / pred.size

    for i in reversed(range(len(params))):
        W, b = params[i]
        a_prev = activations[i]                 # what fed into this layer

        dW = a_prev.T @ delta                   # gradient for weights
        db = delta.sum(axis=0)                  # gradient for biases
        grads[i] = (dW, db)

        if i > 0:
            # Pass the error back to the previous layer.
            # (1 - a^2) is the derivative of tanh.
            delta = (delta @ W.T) * (1 - a_prev ** 2)

    return grads


def train_adam(sizes, X_tr, Y_tr, X_va=None, Y_va=None, lr=0.003, epochs=3000, seed=42):
    local_rng = np.random.default_rng(seed)
    p = init_weights(sizes, local_rng)
    m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in p]
    v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in p]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    train_hist, val_hist = [], []
    for epoch in range(1, epochs + 1):
        pred, acts = forward_with_cache(X_tr, p)
        train_hist.append(mse_loss(pred, Y_tr))
        if X_va is not None:
            val_hist.append(mse_loss(forward(X_va, p), Y_va))

        grads = backward(pred, Y_tr, acts, p)
        new_p = []
        for i, ((W, b), (dW, db)) in enumerate(zip(p, grads)):
            m[i][0] = beta1 * m[i][0] + (1 - beta1) * dW
            m[i][1] = beta1 * m[i][1] + (1 - beta1) * db
            v[i][0] = beta2 * v[i][0] + (1 - beta2) * (dW ** 2)
            v[i][1] = beta2 * v[i][1] + (1 - beta2) * (db ** 2)

            mW_hat = m[i][0] / (1 - beta1 ** epoch)
            mb_hat = m[i][1] / (1 - beta1 ** epoch)
            vW_hat = v[i][0] / (1 - beta2 ** epoch)
            vb_hat = v[i][1] / (1 - beta2 ** epoch)

            W = W - lr * mW_hat / (np.sqrt(vW_hat) + eps)
            b = b - lr * mb_hat / (np.sqrt(vb_hat) + eps)
            new_p.append((W, b))
        p = new_p
    return p, train_hist, val_hist


def main():
    # ---- GRADIENT CHECK ----
    # Compare backprop's gradient for one weight with a slow numerical estimate.
    # If they match, backprop is implemented correctly.
    test_params = init_weights(layer_sizes, np.random.default_rng(42))
    pred, acts = forward_with_cache(X_train, test_params)
    grads = backward(pred, Y_train, acts, test_params)

    W0 = test_params[0][0]
    eps = 1e-5
    W0[0, 0] += eps
    loss_plus = mse_loss(forward(X_train, test_params), Y_train)
    W0[0, 0] -= 2 * eps
    loss_minus = mse_loss(forward(X_train, test_params), Y_train)
    W0[0, 0] += eps                                  # put it back
    numerical = (loss_plus - loss_minus) / (2 * eps)
    print(f"Gradient check -> backprop: {grads[0][0][0, 0]:.10f} | numerical: {numerical:.10f}")

    # ===== STEP 6: TRAINING LOOP =====
    lr = 0.003
    epochs = 3000
    params, train_hist, val_hist = train_adam(
        layer_sizes, X_train, Y_train, X_val, Y_val, lr=lr, epochs=epochs, seed=42
    )
    print(f"\nChosen MLP {layer_sizes} (Adam, lr={lr}, {epochs} epochs):")
    print(f"  Final train MSE: {train_hist[-1]:.6f} rad^2 (RMSE {np.degrees(np.sqrt(train_hist[-1])):.3f} deg)")
    print(f"  Final val   MSE: {val_hist[-1]:.6f} rad^2 (RMSE {np.degrees(np.sqrt(val_hist[-1])):.3f} deg)")

    final_pred_rad = forward(X, params)
    final_pred_deg = np.rad2deg(final_pred_rad)
    seam_jump_deg = np.max(np.abs(final_pred_deg[0] - final_pred_deg[-1]))
    target_seam_deg = np.max(np.abs(np.rad2deg(Y[0] - Y[-1])))
    print(f"  Wrap-around max step (frame 299 -> 0): pred {seam_jump_deg:.3f} deg | target {target_seam_deg:.3f} deg")

    np.savetxt(os.path.join(NEURAL_DIR, "predicted_gait.csv"), final_pred_deg, delimiter=",")
    np.savetxt(os.path.join(NEURAL_DIR, "predicted_gait_rad.csv"), final_pred_rad, delimiter=",")

    model_path = os.path.join(NEURAL_DIR, "trained_model.npz")
    model_data = {"layer_sizes": np.array(layer_sizes), "n_frames": np.array(n_frames)}
    for idx, (W, b) in enumerate(params):
        model_data[f"W{idx}"] = W
        model_data[f"b{idx}"] = b
    np.savez(model_path, **model_data)
    print(f"  Saved trained model weights to {model_path}")

    if os.path.isdir(WEBOTS_DIR):
        webots_pred = os.path.join(WEBOTS_DIR, "predicted_gait.csv")
        np.savetxt(webots_pred, final_pred_deg, delimiter=",")
        print(f"  Exported predicted gait to {webots_pred}")

    # ===== STEP 7: PLOTS =====
    joint_labels = [
        f"{leg} {j}"
        for leg in ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
        for j in ["coxa", "femur", "tibia"]
    ]

    # ---- Plot 1: loss curve ----
    plt.figure(figsize=(7, 4))
    plt.plot(train_hist, label="Train MSE (even frames)")
    plt.plot(val_hist, "--", label="Validation MSE (odd frames)")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (rad^2, log scale)")
    plt.title(f"NN Training & Validation Loss ({layer_sizes}, Adam lr={lr})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(NEURAL_DIR, "loss_curve.png"), dpi=150)

    # ---- Plot 2: predicted vs target angles for a few joints ----
    joints_to_show = [0, 1, 2, 21]   # L1 coxa, L1 femur, L1 tibia, R1 coxa
    fig, axes = plt.subplots(len(joints_to_show), 1, figsize=(7, 8), sharex=True)
    for ax, j in zip(axes, joints_to_show):
        ax.plot(t, np.rad2deg(Y[:, j]), label="GA Target")
        ax.plot(t, final_pred_deg[:, j], "--", label="NN Predicted")
        ax.set_ylabel("deg")
        ax.set_title(joint_labels[j])
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    plt.savefig(os.path.join(NEURAL_DIR, "predicted_vs_target.png"), dpi=150)

    key_steps = [0, 75, 150, 225]
    fig = plt.figure(figsize=(3.2 * len(key_steps), 3.6))
    for k, step in enumerate(key_steps):
        ax = fig.add_subplot(1, len(key_steps), k + 1, projection="3d")
        viz._setup_axes(ax)
        lines = viz._make_leg_lines(ax)
        viz._update_leg_lines(lines, final_pred_deg[step].reshape(config.n_legs, 3))
        swing = [fit.LEG_NAMES[i] for i in np.where(fit.SWING[k])[0]]
        ax.set_title(f"NN frame {step}/300\nswing: {', '.join(swing) or '-'}", fontsize=8)
    fig.suptitle("NN Predicted 3D Spider Poses (Key Gait Phases)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(NEURAL_DIR, "predicted_poses.png"), dpi=150)

    # ---- Error per joint (console summary) ----
    per_joint_mse = np.mean((final_pred_rad - Y) ** 2, axis=0)
    worst = int(np.argmax(per_joint_mse))
    print(f"  Worst joint: {joint_labels[worst]} (MSE {per_joint_mse[worst]:.6f} rad^2, "
          f"RMSE {np.degrees(np.sqrt(per_joint_mse[worst])):.3f} deg)")

    # ===== STEP 8: ARCHITECTURE COMPARISON =====
    architectures = [
        [2, 4, 24],               # tiny: 1 hidden layer, 4 neurons
        [2, 16, 24],              # small: 1 hidden layer
        [2, 64, 24],              # medium: 1 hidden layer
        [2, 64, 64, 24],          # chosen: 2 hidden layers
        [2, 128, 128, 128, 24],   # large: 3 hidden layers
    ]

    plt.figure(figsize=(8, 5))
    print(f"\n{'Architecture':<26}{'Params':>8}{'Train MSE':>14}{'Val MSE':>14}")
    for sizes in architectures:
        p_arch, th, vh = train_adam(sizes, X_train, Y_train, X_val, Y_val, lr=lr, epochs=epochs, seed=42)
        n_params = sum(W.size + b.size for W, b in p_arch)
        print(f"{str(sizes):<26}{n_params:>8}{th[-1]:>14.6f}{vh[-1]:>14.6f}")
        plt.plot(vh, label=f"{sizes} ({n_params} params)")

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE (rad^2, log scale)")
    plt.title(f"Architecture comparison on GA gait (Adam lr={lr}, {epochs} epochs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(NEURAL_DIR, "architecture_comparison.png"), dpi=150)
    print("saved trained_model.npz, loss_curve.png, predicted_vs_target.png, predicted_poses.png, architecture_comparison.png, predicted_gait.csv in neural/")

    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
