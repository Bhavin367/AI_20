import os
import numpy as np
import matplotlib.pyplot as plt

import nntrainingmethod as nnt     # teammate's network, backprop and optimisers
import chrom_encode as ce


# ---------------------------------------------------------------------------
# Part 1: evidence that backpropagation is correct (gradient check)
# ---------------------------------------------------------------------------
# backward() in nntrainingmethod.py calculates the gradient of the loss with
# respect to every weight using the chain rule:
#   - output layer: dLoss/dPred = 2 * (pred - Y) / N   (derivative of MSE)
#   - each layer:   dW = (input to the layer)^T @ delta,  db = sum of delta
#   - going back a layer, delta is multiplied by W^T and by the derivative
#     of tanh, which is 1 - tanh(x)^2 = 1 - a^2
#
# A falling loss does not prove those formulas are right; a wrong gradient
# can still sometimes reduce the loss. So we check them a second, completely
# independent way: numerically. For a weight w we nudge it up and down by a
# tiny amount e and measure how the loss changes:
#
#   numerical gradient = (Loss(w + e) - Loss(w - e)) / (2e)
#
# This needs no calculus at all. If backprop is correct, the two gradients
# agree. We compare them with the relative error
#
#   |numerical - backprop| / (|numerical| + |backprop|)
#
#   < 1e-6  -> backprop is correct
#   > 1e-3  -> there is a bug in backward()
def gradient_check(n_samples=20, checks_per_param=25, eps=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    params = nnt.init_weights(nnt.layer_sizes, rng)
    X, Y = nnt.X[:n_samples], nnt.Y[:n_samples]

    def loss():
        return nnt.mse_loss(nnt.forward_with_cache(X, params)[0], Y)

    pred, acts = nnt.forward_with_cache(X, params)
    grads = nnt.backward(pred, Y, acts, params)

    print("Gradient check (backprop vs numerical gradient)")
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
                p[idx] = original                          # put the weight back
                numerical = (loss_up - loss_down) / (2 * eps)
                error = abs(numerical - g[idx]) / max(abs(numerical) + abs(g[idx]), 1e-12)
                worst = max(worst, error)
            worst_overall = max(worst_overall, worst)
            print(f"  layer {i + 1} {name:<6}{str(p.shape):<12}{worst:>22.2e}")
    verdict = "PASS - backprop is correct" if worst_overall < 1e-5 else "FAIL - check backward()"
    print(f"  worst overall: {worst_overall:.2e}  -> {verdict}\n")
    return worst_overall


# ---------------------------------------------------------------------------
# Part 2: the evolved gait as training data
# ---------------------------------------------------------------------------
# The GA evolves a gait of 4 key frames (96 genes = 4 frames x 24 angles).
# The network takes the phase of the walk cycle as input (sin, cos of the
# angle around the cycle, so the end of the cycle joins smoothly back to
# the start) and must output the 24 joint angles for that moment.
#
# To get 300 training frames from 4 key frames we interpolate in a loop:
# frame 0 sits at phase 0, frame 1 at 1/4 of the cycle, and so on, with
# frame 3 blending back into frame 0. Angles are converted to radians so
# targets are around +-1.5, a good range for the network.
#
# The GA's best genome should be saved with np.save("gait.npy", best_genome).
# If gait.npy does not exist yet, the teammate's placeholder data is used.
def gait_targets(genome, n_frames=len(nnt.X)):
    keyframes = np.deg2rad(ce.to_frames(genome).reshape(ce.N_FRAMES, -1))   # (4, 24)
    key_phase = np.arange(ce.N_FRAMES + 1) / ce.N_FRAMES                    # 0, .25, .5, .75, 1
    looped = np.vstack([keyframes, keyframes[:1]])                          # frame 0 again at the end
    phase = np.arange(n_frames) / n_frames
    return np.column_stack([np.interp(phase, key_phase, looped[:, j]) for j in range(looped.shape[1])])


def load_data():
    if os.path.exists("gait.npy"):
        print("Training on the evolved gait from gait.npy\n")
        return nnt.X, gait_targets(np.load("gait.npy"))
    print("gait.npy not found - training on the placeholder data instead\n")
    return nnt.X, nnt.Y


# ---------------------------------------------------------------------------
# Part 3: convergence
# ---------------------------------------------------------------------------
# Same training as nntrainingmethod.py (full batch, one update per epoch)
# but on any X, Y, so it can be run on the gait.
def train_on(X, Y, optimizer, epochs=3000, seed=42):
    rng = np.random.default_rng(seed)
    params = nnt.init_weights(nnt.layer_sizes, rng)
    history = []
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(epochs):
            pred, acts = nnt.forward_with_cache(X, params)
            history.append(nnt.mse_loss(pred, Y))
            params = optimizer.step(params, nnt.backward(pred, Y, acts, params))
    return history, params


if __name__ == "__main__":
    gradient_check()
    X, Y = load_data()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ---- Learning rate, with diverged runs kept off the graph ----
    # A learning rate that is too big overshoots the minimum every step, so
    # the loss explodes to infinity. Plotting infinity squashes every other
    # line flat, so diverged runs are listed in the legend title instead.
    print("Learning rate  (gradient descent, full batch)")
    diverged = []
    for lr in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        h, _ = train_on(X, Y, nnt.GradientDescent(lr))
        h = np.array(h)
        if not np.isfinite(h[-1]) or h[-1] > h[0]:
            diverged.append(str(lr))
            print(f"  lr = {lr:<6} diverged")
            continue
        print(f"  lr = {lr:<6} final loss {h[-1]:.7f}")
        axes[0].plot(h, label=f"lr = {lr}")
    title = "diverged: lr = " + ", ".join(diverged) if diverged else None
    axes[0].legend(fontsize=8, title=title, title_fontsize=8)
    axes[0].set_title("Learning rate (GD, full batch)")

    # ---- Best setup: Adam, trained to convergence ----
    print("\nAdam (lr = 0.001)")
    h, params = train_on(X, Y, nnt.Adam(0.001))
    print(f"  start loss {h[0]:.5f} -> final loss {h[-1]:.7f} "
          f"({h[0] / h[-1]:.0f}x lower)")
    axes[1].plot(h, color="tab:green")
    axes[1].set_title("Convergence (Adam, lr = 0.001)")

    # ---- What the network learned vs the target ----
    # If training converged, the predicted joint angles lie on top of the
    # target angles across the whole walk cycle.
    pred = nnt.forward_with_cache(X, params)[0]
    for j, name in [(0, "L1 coxa"), (1, "L1 femur"), (2, "L1 tibia")]:
        line, = axes[2].plot(np.rad2deg(Y[:, j]), linewidth=3, alpha=0.35)
        axes[2].plot(np.rad2deg(pred[:, j]), "--", color=line.get_color(), label=name)
    axes[2].set_title("Target (thick) vs network (dashed)")
    axes[2].set_xlabel("Frame in walk cycle")
    axes[2].set_ylabel("Angle (degrees)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    for ax in axes[:2]:
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss (log scale)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("backprop_convergence.png", dpi=150)
    plt.show()
