"""
nntrainingmethod.py - how the network is trained.

    1. Optimisers      : gradient descent, momentum, Adam
    2. train()         : the one training loop used everywhere
    3. run_experiments : compare architecture, activation, learning rate,
                         batch size and optimiser on the evolved gait

Nothing runs on import. Use main.py.
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

import neuralnetwork as nn

# Fixed colour order, so lines stay distinguishable (colour-blind checked)
COLOURS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17"]


# ===========================================================================
# 1. OPTIMISERS
# ===========================================================================
# Each optimiser has step(params, grads) and returns the updated weights.
# Only the update rule differs.

class GradientDescent:
    """w = w - lr * gradient"""
    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        return [(W - self.lr * dW, b - self.lr * db)
                for (W, b), (dW, db) in zip(params, grads)]


class Momentum:
    """Keeps a running 'velocity' of past gradients, so steps speed up in
    directions that stay consistent and wobble less where they keep flipping."""
    def __init__(self, lr, beta=0.9):
        self.lr, self.beta = lr, beta
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
        self.velocity = [(self.beta * vW + dW, self.beta * vb + db)
                         for (vW, vb), (dW, db) in zip(self.velocity, grads)]
        return [(W - self.lr * vW, b - self.lr * vb)
                for (W, b), (vW, vb) in zip(params, self.velocity)]


class Adam:
    """Momentum (m) plus a per-weight step size scaled by recent gradient size
    (v). m_hat and v_hat correct the bias from starting m and v at zero."""
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


OPTIMISERS = {"gd": GradientDescent, "momentum": Momentum, "adam": Adam}


# ===========================================================================
# 2. TRAINING LOOP
# ===========================================================================
def train(X, Y, layers, activation="tanh", optimiser="adam", lr=0.001,
          batch_size=None, epochs=3000, seed=42):
    """Train a new network and return (loss_history, params).

    batch_size=None is full batch: one update per epoch using every sample.
    Otherwise the samples are shuffled each epoch and there is one update per
    mini-batch. The loss is always measured on all the training samples.
    loss_history[0] is the loss before any training.
    """
    rng = np.random.default_rng(seed)            # fixed seed: results repeat
    params = nn.init_weights(layers, rng, activation)
    opt = OPTIMISERS[optimiser](lr)
    n = len(X)
    bs = n if batch_size is None else batch_size

    history = [nn.mse_loss(nn.predict(X, params, activation), Y)]
    with np.errstate(over="ignore", invalid="ignore"):   # a too-high lr can overflow
        for _ in range(epochs):
            order = rng.permutation(n) if bs < n else np.arange(n)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                pred, acts = nn.forward_with_cache(X[idx], params, activation)
                grads = nn.backward(pred, Y[idx], acts, params, activation)
                params = opt.step(params, grads)
            history.append(nn.mse_loss(nn.predict(X, params, activation), Y))
    return history, params


# ===========================================================================
# 3. EXPERIMENTS
# ===========================================================================
# Each experiment changes ONE setting and keeps the rest equal to `base`
# (the final configuration in main.py).
#
# Fairness notes for the write-up:
#   - Smaller batches make more weight updates per epoch (batch 8 on 300
#     samples is 38 updates, full batch is 1), so compare the update counts.
#   - Results use one random seed. Small differences may not be meaningful.

def experiment_list(base):
    """(title, [(label, settings changed from base), ...]) for each experiment."""
    n_in, n_out = base["layers"][0], base["layers"][-1]
    hidden = [[], [8], [64], [64, 64], [128, 128, 128]]   # [] = linear model
    return [
        ("Architecture",
         [(str([n_in, *h, n_out]), {"layers": [n_in, *h, n_out]}) for h in hidden]),
        ("Activation function",
         [(name, {"activation": name}) for name in ["tanh", "relu", "sigmoid"]]),
        ("Learning rate (gradient descent)",
         [(f"lr = {lr}", {"optimiser": "gd", "lr": lr, "batch_size": None})
          for lr in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]]),
        ("Learning rate (Adam)",
         [(f"lr = {lr}", {"optimiser": "adam", "lr": lr})
          for lr in [0.0001, 0.001, 0.01, 0.1]]),
        ("Batch size",
         [("full" if bs is None else str(bs), {"batch_size": bs})
          for bs in [None, 64, 32, 8]]),
        ("Optimiser",
         [(label, {"optimiser": name, "lr": lr})
          for label, name, lr in [("GD 0.1", "gd", 0.1), ("Momentum 0.1", "momentum", 0.1),
                                  ("Adam 0.001", "adam", 0.001), ("Adam 0.01", "adam", 0.01)]]),
    ]


def run_experiments(X, Y, base, out_dir):
    """Run every experiment; save a table (CSV) and one figure of loss curves."""
    experiments = experiment_list(base)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    rows = []

    for ax, (title, runs) in zip(axes.flat, experiments):
        print(f"\n  {title}")
        print(f"    {'setting':<26}{'params':>8}{'updates':>10}{'final loss':>14}")
        diverged = []
        for i, (label, changes) in enumerate(runs):
            settings = {**base, **changes}
            history, params = train(X, Y, **settings)
            bs = settings["batch_size"] or len(X)
            updates = settings["epochs"] * int(np.ceil(len(X) / bs))
            n_params = nn.count_parameters(params)

            # Diverged: the loss exploded or ended worse than it started
            if not np.isfinite(history[-1]) or history[-1] > history[0]:
                diverged.append(label)
                result = "diverged"
            else:
                result = f"{history[-1]:.7f}"
                ax.plot(history, color=COLOURS[i], linewidth=1.5, label=label)
            print(f"    {label:<26}{n_params:>8}{updates:>10}{result:>14}")
            rows.append([title, label, n_params, updates, result])

        legend_title = "diverged: " + ", ".join(diverged) if diverged else None
        ax.legend(fontsize=8, title=legend_title, title_fontsize=8)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss (log scale)")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "experiments.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "experiments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "setting", "parameters", "weight_updates", "final_loss"])
        writer.writerows(rows)
