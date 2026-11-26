"""
neuralnetwork.py - the data and the network (written from scratch with NumPy).

    1. Data          : load the evolved gait from gait.csv and build X, Y
    2. Network       : weights, activation functions, forward pass
    3. Loss          : mean squared error
    4. Backpropagation
    5. Saving and loading the trained model

Nothing runs on import. Use main.py.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GAIT_FILE = HERE / "gait.csv"

N_SAMPLES = 300      # training samples spread over one walk cycle
N_JOINTS = 24        # 8 legs x 3 joints

# Order of the 24 values in each frame. This must match the GA encoding.
LEG_NAMES = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
JOINT_TYPES = ["coxa", "femur", "tibia"]
JOINT_NAMES = [f"{leg} {joint}" for leg in LEG_NAMES for joint in JOINT_TYPES]


# ===========================================================================
# 1. DATA
# ===========================================================================
# INPUT  : the phase of the walk cycle, written as (sin, cos). The end of the
#          cycle then sits next to the start, so the network sees a loop.
# OUTPUT : the 24 joint angles (radians) for that moment of the cycle.
#
# The GA gives a few key frames (4 in our encoding). To get more training
# samples we interpolate linearly between neighbouring key frames, with the
# last frame blending back into the first.

def load_keyframes(path=GAIT_FILE):
    """Read the gait file and return an array (n_frames, 24) in DEGREES.

    Accepts one frame per row, or all genes in a single row or column
    (stored frame after frame). A missing file is an error on purpose, so the
    network can never train on the wrong data without anyone noticing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Gait file not found: {path}\n"
                                "Put gait.csv in the same folder as these scripts.")
    try:
        G = np.loadtxt(path, delimiter=",", ndmin=2)
    except ValueError as err:
        raise ValueError(f"Could not read {path.name}. It must contain only numbers "
                         f"separated by commas, with no header row.\nDetails: {err}")

    # A flat file: one row or one column holding every gene
    if 1 in G.shape and G.size > N_JOINTS and G.size % N_JOINTS == 0:
        print(f"{path.name} is flat {G.shape}; reshaping to ({G.size // N_JOINTS}, {N_JOINTS})")
        G = G.reshape(-1, N_JOINTS)

    if G.shape[1] != N_JOINTS or len(G) < 2:
        raise ValueError(f"{path.name} has shape {G.shape}; expected (n_frames, {N_JOINTS}) "
                         "with at least 2 frames. A mirrored 48-gene genome must be "
                         "expanded to the full 8 legs first.")
    if not np.all(np.isfinite(G)):
        raise ValueError(f"{path.name} contains NaN or infinite values.")
    if np.max(np.abs(G)) <= 2 * np.pi:
        print("WARNING: every angle is within +-6.3, so the file may be in radians. "
              "This code expects degrees.")
    return G


def phase_to_input(phase):
    """phase is the fraction of the walk cycle (0 to 1). Returns (n, 2)."""
    angle = 2 * np.pi * np.asarray(phase)
    return np.column_stack([np.sin(angle), np.cos(angle)])


def interpolate_gait(keyframes, phase):
    """Linear interpolation between key frames, looping back to frame 0."""
    n_frames = len(keyframes)
    key_phase = np.arange(n_frames + 1) / n_frames          # e.g. 0, .25, .5, .75, 1
    looped = np.vstack([keyframes, keyframes[:1]])          # frame 0 again at the end
    return np.column_stack([np.interp(phase, key_phase, looped[:, j])
                            for j in range(keyframes.shape[1])])


def make_dataset(keyframes_deg, n_samples=N_SAMPLES, offset=0.0):
    """Build (X, Y, phase). Y is in radians.

    offset shifts the sample points by a fraction of one step. offset=0.5
    gives the points halfway between the training samples, used for testing.
    """
    phase = (np.arange(n_samples) + offset) / n_samples
    Y = interpolate_gait(np.deg2rad(keyframes_deg), phase)
    return phase_to_input(phase), Y, phase


# ===========================================================================
# 2. NETWORK
# ===========================================================================
# Each activation is stored as (function, derivative). The derivative is
# written in terms of the activation's OUTPUT a, which backprop already has.
ACTIVATIONS = {
    "tanh":    (np.tanh,                           lambda a: 1 - a ** 2),
    "relu":    (lambda z: np.maximum(z, 0),        lambda a: (a > 0).astype(float)),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)),    lambda a: a * (1 - a)),
}


def init_weights(sizes, rng, activation="tanh"):
    """One (W, b) pair per layer. W has shape (neurons in, neurons out).

    Weights start small and random, scaled by 1/sqrt(inputs) so signals do not
    blow up or vanish. ReLU uses sqrt(2/inputs) because it zeroes about half
    of its inputs. Biases start at zero.
    """
    gain = np.sqrt(2) if activation == "relu" else 1.0
    return [(rng.normal(0, gain / np.sqrt(n_in), size=(n_in, n_out)), np.zeros(n_out))
            for n_in, n_out in zip(sizes[:-1], sizes[1:])]


def forward_with_cache(X, params, activation="tanh"):
    """Forward pass. Returns the prediction and each layer's input, which
    backprop needs.

    Hidden layers: weighted sum, then the activation function.
    Output layer : weighted sum only, because joint angles can be any value.
    """
    f = ACTIVATIONS[activation][0]
    activations = [X]
    a = X
    for W, b in params[:-1]:
        a = f(a @ W + b)
        activations.append(a)
    W, b = params[-1]
    return a @ W + b, activations


def predict(X, params, activation="tanh"):
    """Joint angles (radians) for inputs X."""
    return forward_with_cache(X, params, activation)[0]


def count_parameters(params):
    return sum(W.size + b.size for W, b in params)


# ===========================================================================
# 3. LOSS
# ===========================================================================
def mse_loss(pred, Y):
    """Mean squared error over every angle in every sample. Suits regression:
    errors cannot cancel out, big mistakes cost more, and the derivative is
    simple. 0 means a perfect match."""
    return np.mean((pred - Y) ** 2)


# ===========================================================================
# 4. BACKPROPAGATION
# ===========================================================================
def backward(pred, Y, activations, params, activation="tanh"):
    """Gradient of the MSE loss for every weight and bias, using the chain
    rule from the output layer back to the first layer."""
    derivative = ACTIVATIONS[activation][1]
    grads = [None] * len(params)

    # dLoss/dPred (derivative of the mean squared error)
    delta = 2 * (pred - Y) / pred.size

    for i in reversed(range(len(params))):
        W, _ = params[i]
        a_prev = activations[i]                      # what fed into this layer
        grads[i] = (a_prev.T @ delta,                # dLoss/dW
                    delta.sum(axis=0))               # dLoss/db
        if i > 0:                                    # pass the error back a layer
            delta = (delta @ W.T) * derivative(a_prev)
    return grads


# ===========================================================================
# 5. SAVING AND LOADING
# ===========================================================================
def save_model(path, params, activation):
    """Save the weights and activation name to one .npz file."""
    arrays = {"activation": activation, "n_layers": len(params)}
    for i, (W, b) in enumerate(params):
        arrays[f"W{i}"] = W
        arrays[f"b{i}"] = b
    np.savez(path, **arrays)


def load_model(path):
    """Return (params, activation) from a file written by save_model()."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No saved model at {path}. Run  python main.py  first.")
    with np.load(path) as data:
        params = [(data[f"W{i}"], data[f"b{i}"]) for i in range(int(data["n_layers"]))]
        return params, str(data["activation"])
