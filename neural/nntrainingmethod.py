import os
import numpy as np
import matplotlib.pyplot as plt

NEURAL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(NEURAL_DIR)


def load_ga_gait_radians():
    rad_path = os.path.join(BASE_DIR, "gait_rad.csv")
    deg_path = os.path.join(BASE_DIR, "gait.csv")
    if os.path.exists(rad_path):
        return np.loadtxt(rad_path, delimiter=",")
    if os.path.exists(deg_path):
        return np.deg2rad(np.loadtxt(deg_path, delimiter=","))
    raise FileNotFoundError(f"Could not find gait_rad.csv or gait.csv in {BASE_DIR}. Run main.py (Part 1) first.")


# ===== DATA (same as nn_scratch.py) =====
n_frames = 300
t = np.arange(n_frames)
phase = 2 * np.pi * t / n_frames
X = np.column_stack([np.sin(phase), np.cos(phase)])                 # (300, 2)
Y = load_ga_gait_radians()                                          # (300, 24)

layer_sizes = [2, 64, 64, 24]   # chosen architecture, tanh hidden layers


# ===== NETWORK (same as nn_scratch.py) =====
def init_weights(sizes, rng):
    """Small random weights scaled by 1/sqrt(inputs), zero biases."""
    return [(rng.normal(0, 1 / np.sqrt(n_in), size=(n_in, n_out)), np.zeros(n_out))
            for n_in, n_out in zip(sizes[:-1], sizes[1:])]


def forward_with_cache(X_in, params):
    activations = [X_in]
    a = X_in
    for W, b in params[:-1]:
        a = np.tanh(a @ W + b)
        activations.append(a)
    W, b = params[-1]
    return a @ W + b, activations


def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)


def backward(pred, target, activations, params):
    grads = [None] * len(params)
    delta = 2 * (pred - target) / pred.size
    for i in reversed(range(len(params))):
        W, b = params[i]
        a_prev = activations[i]
        grads[i] = (a_prev.T @ delta, delta.sum(axis=0))
        if i > 0:
            delta = (delta @ W.T) * (1 - a_prev ** 2)
    return grads


# ===== OPTIMISERS =====
# Each optimiser has a step() that takes the weights and their gradients
# and returns the updated weights. Only the update rule differs.

class GradientDescent:
    """Plain gradient descent: w = w - lr * gradient."""
    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        return [(W - self.lr * dW, b - self.lr * db)
                for (W, b), (dW, db) in zip(params, grads)]


class Momentum:
    """Gradient descent with momentum. Keeps a running 'velocity' of past
    gradients, so steps speed up in directions that stay consistent and
    wobble less in directions that keep flipping."""
    def __init__(self, lr, beta=0.9):
        self.lr, self.beta = lr, beta
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
        new_params, new_velocity = [], []
        for (W, b), (dW, db), (vW, vb) in zip(params, grads, self.velocity):
            vW = self.beta * vW + dW
            vb = self.beta * vb + db
            new_params.append((W - self.lr * vW, b - self.lr * vb))
            new_velocity.append((vW, vb))
        self.velocity = new_velocity
        return new_params


class Adam:
    """Adam: momentum (m) plus a per-weight step size scaled by recent
    gradient size (v). Weights with small gradients get relatively bigger
    steps. m_hat and v_hat correct the bias from starting m and v at zero."""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [[np.zeros_like(p) for p in layer] for layer in params]
            self.v = [[np.zeros_like(p) for p in layer] for layer in params]
        self.t += 1
        new_params = []
        for i, (layer, layer_grads) in enumerate(zip(params, grads)):
            new_layer = []
            for k, (p, g) in enumerate(zip(layer, layer_grads)):
                self.m[i][k] = self.beta1 * self.m[i][k] + (1 - self.beta1) * g
                self.v[i][k] = self.beta2 * self.v[i][k] + (1 - self.beta2) * g ** 2
                m_hat = self.m[i][k] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i][k] / (1 - self.beta2 ** self.t)
                new_layer.append(p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
            new_params.append(tuple(new_layer))
        return new_params


# ===== TRAINING LOOP (now supports mini-batches) =====
def train(optimizer, batch_size=None, epochs=3000, seed=42):
    rng = np.random.default_rng(seed)
    params = init_weights(layer_sizes, rng)
    n = len(X)
    bs = n if batch_size is None else batch_size
    history = []
    with np.errstate(over="ignore", invalid="ignore"):   # a too-high lr can overflow
        for _ in range(epochs):
            order = rng.permutation(n) if bs < n else np.arange(n)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                pred, acts = forward_with_cache(X[idx], params)
                grads = backward(pred, Y[idx], acts, params)
                params = optimizer.step(params, grads)
            history.append(mse_loss(forward_with_cache(X, params)[0], Y))
    return history


def report(label, history):
    final = history[-1]
    text = "diverged (nan)" if not np.isfinite(final) else f"{final:.7f}"
    print(f"  {label:<22}{text}")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ---- Experiment 1: learning rate (plain GD, full batch) ----
    print("1. Learning rate (gradient descent, full batch)")
    for lr in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        h = np.array(train(GradientDescent(lr)), dtype=float)
        report(f"lr = {lr}", h)
        h_plot = np.where(np.isfinite(h) & (h <= 10.0), h, np.nan)
        axes[0].plot(h_plot, label=f"lr = {lr}")
    axes[0].set_ylim(5e-5, 2.0)
    axes[0].set_title("Learning rate (GD, full batch)")

    # ---- Experiment 2: batch size (plain GD, lr 0.1) ----
    print("\n2. Batch size (gradient descent, lr = 0.1)")
    for bs in [None, 64, 32, 8]:
        label = "full (300)" if bs is None else str(bs)
        h = train(GradientDescent(0.1), batch_size=bs)
        report(f"batch = {label}", h)
        axes[1].plot(h, label=f"batch = {label}")
    axes[1].set_title("Batch size (GD, lr = 0.1)")

    # ---- Experiment 3: optimiser (full batch) ----
    print("\n3. Optimiser (full batch)")
    for name, opt in [("Gradient descent 0.1", GradientDescent(0.1)),
                      ("Momentum 0.1", Momentum(0.1)),
                      ("Adam 0.001", Adam(0.001)),
                      ("Adam 0.003", Adam(0.003))]:
        h = train(opt)
        report(name, h)
        axes[2].plot(h, label=name)
    axes[2].set_title("Optimiser (full batch)")

    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss (rad^2, log scale)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(NEURAL_DIR, "training_comparison.png"), dpi=150)
    print("saved training_comparison.png in neural/")

    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
