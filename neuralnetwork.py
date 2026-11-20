import numpy as np

rng = np.random.default_rng(42)  # fixed seed so results repeat

# ===== STEP 1: TRAINING DATA =====
# 300 frames in one walk cycle. Each frame becomes 2 numbers (sin, cos),
# so frame 299 sits right next to frame 0: the network learns the walk loops.
n_frames = 300
t = np.arange(n_frames)
phase = 2 * np.pi * t / n_frames
X = np.column_stack([np.sin(phase), np.cos(phase)])   # inputs, shape (300, 2)

# Real gait from our GA (Part 1): 300 frames x 24 joint angles.
# gait.csv is in degrees, so convert to radians.
Y = np.radians(np.loadtxt("gait.csv", delimiter=","))

print("X shape:", X.shape)
print("Y shape:", Y.shape)


# ===== STEP 2: ARCHITECTURE / WEIGHTS =====
# 2 inputs -> 64 -> 64 -> 24 outputs
layer_sizes = [2, 64, 64, 24]

def init_weights(sizes):
    """One weight matrix W and bias vector b per layer.
    W has shape (neurons in, neurons out). Starting values are small and
    random, scaled by 1/sqrt(inputs), so signals don't blow up or vanish."""
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        W = rng.normal(0, 1 / np.sqrt(n_in), size=(n_in, n_out))
        b = np.zeros(n_out)
        params.append((W, b))
    return params

params = init_weights(layer_sizes)

for i, (W, b) in enumerate(params):
    print(f"Layer {i+1}: W {W.shape}, b {b.shape}")
print("Total parameters:", sum(W.size + b.size for W, b in params))


# ===== STEP 3: FORWARD PASS =====
def forward(X, params):
    """Push inputs through the network.
    Hidden layers: weighted sum, then tanh (smooth, outputs between -1 and 1).
    Output layer: weighted sum only (no activation), since angles can be any value."""
    a = X
    for W, b in params[:-1]:
        a = np.tanh(a @ W + b)       # hidden layer
    W, b = params[-1]
    return a @ W + b                 # output layer: 24 angles

pred = forward(X, params)
print("Prediction shape:", pred.shape)
print("First frame prediction:", np.round(pred[0], 2))

# ===== STEP 4: LOSS =====
def mse_loss(pred, Y):
    """Mean squared error: average of (prediction - target)^2 over every
    angle in every frame. Squaring makes all errors positive and punishes
    big mistakes more than small ones. 0 = perfect match."""
    return np.mean((pred - Y) ** 2)

print("Starting loss:", mse_loss(pred, Y))

# ===== STEP 5: BACKPROPAGATION =====
def forward_with_cache(X, params):
    """Same as forward(), but also keeps each layer's output,
    because backprop needs them to compute gradients."""
    activations = [X]             # activations[0] = the input
    a = X
    for W, b in params[:-1]:
        a = np.tanh(a @ W + b)
        activations.append(a)     # output of each hidden layer
    W, b = params[-1]
    pred = a @ W + b
    return pred, activations


def backward(pred, Y, activations, params):
    """Work out the gradient of the MSE loss for every weight and bias,
    moving backwards from the output layer to the first layer."""
    grads = [None] * len(params)

    # How the loss changes with each prediction (derivative of MSE)
    delta = 2 * (pred - Y) / pred.size

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


# ---- GRADIENT CHECK ----
# Compare backprop's gradient for one weight with a slow numerical estimate.
# If they match, backprop is implemented correctly.
pred, acts = forward_with_cache(X, params)
grads = backward(pred, Y, acts, params)

W0 = params[0][0]
eps = 1e-5
W0[0, 0] += eps
loss_plus = mse_loss(forward(X, params), Y)
W0[0, 0] -= 2 * eps
loss_minus = mse_loss(forward(X, params), Y)
W0[0, 0] += eps                                  # put it back

numerical = (loss_plus - loss_minus) / (2 * eps)
print("Backprop gradient: ", grads[0][0][0, 0])
print("Numerical gradient:", numerical)

# ===== STEP 6: TRAINING LOOP =====
learning_rate = 0.1     # step size for each weight update
epochs = 3000           # number of passes over all 300 frames

params = init_weights(layer_sizes)   # fresh, untrained network
loss_history = []                    # saved so we can plot it later

for epoch in range(epochs):
    # 1. Forward pass
    pred, acts = forward_with_cache(X, params)

    # 2. Loss
    loss = mse_loss(pred, Y)
    loss_history.append(loss)

    # 3. Backprop
    grads = backward(pred, Y, acts, params)

    # 4. Update: plain gradient descent, w = w - learning_rate * gradient
    params = [(W - learning_rate * dW, b - learning_rate * db)
              for (W, b), (dW, db) in zip(params, grads)]

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d}   loss {loss:.6f}")

print("Final loss:", loss_history[-1])

# ---- Save the network ----
save_data = {}
for i, (W, b) in enumerate(params):   # go through each layer
    save_data[f"W{i}"] = W            # the weights
    save_data[f"b{i}"] = b            # the biases
np.savez("trained_model.npz", **save_data)   # put them all in one file
print("Saved model to trained_model.npz")

# ===== STEP 7: PLOTS =====
import matplotlib.pyplot as plt

# Predictions from the trained network
final_pred = forward(X, params)

joint_labels = [f"{leg} {j}" for leg in ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
                for j in ["coxa", "femur", "tibia"]]

# ---- Plot 1: loss curve ----
# Log scale on the y-axis, because the loss shrinks by several
# orders of magnitude and a normal scale would hide the later progress.
plt.figure(figsize=(7, 4))
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE loss (log scale)")
plt.title(f"Training loss  (layers {layer_sizes}, lr = {learning_rate})")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)

# ---- Plot 2: predicted vs target angles for a few joints ----
# Solid line = target from the gait, dashed = network's prediction.
# If they overlap, the network has learned that joint's motion.
joints_to_show = [0, 1, 2, 21]   # L1 coxa, L1 femur, L1 tibia, R1 coxa
fig, axes = plt.subplots(len(joints_to_show), 1, figsize=(7, 8), sharex=True)
for ax, j in zip(axes, joints_to_show):
    ax.plot(t, Y[:, j], label="Target")
    ax.plot(t, final_pred[:, j], "--", label="Predicted")
    ax.set_ylabel("rad")
    ax.set_title(joint_labels[j])
    ax.grid(True, alpha=0.3)
axes[0].legend()
axes[-1].set_xlabel("Frame")
plt.tight_layout()
plt.savefig("predicted_vs_target.png", dpi=150)

# ---- Error per joint (console summary) ----
per_joint_error = np.mean((final_pred - Y) ** 2, axis=0)
worst = np.argmax(per_joint_error)
print(f"Worst joint: {joint_labels[worst]}  (MSE {per_joint_error[worst]:.6f})")

# ---- Show one frame: real angles vs network angles ----
frame = 150   # pick a frame (midway through the walk)
print(f"\nFrame {frame} (degrees):")
print("Joint       Target  Predicted  Error")
for j in range(24):   # go through all 24 joints
    target_deg = np.degrees(Y[frame, j])          # real angle
    pred_deg = np.degrees(final_pred[frame, j])   # network's angle
    print(f"{joint_labels[j]:<10} {target_deg:7.1f} {pred_deg:9.1f} {pred_deg - target_deg:6.1f}")

# ---- Draw the spiders ----
import viz   # file that draws the spider
frames_to_show = [0, 75, 150, 225]   # 4 bits of the walk
fig = plt.figure(figsize=(12, 6))    # Make a big picture
for col, f in enumerate(frames_to_show):
    for row, (name, data) in enumerate([("Target", Y), ("Predicted", final_pred)]):
        ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection="3d")   # make a 3D box
        viz._setup_axes(ax)                       # draw body and floor
        lines = viz._make_leg_lines(ax)           # make the legs
        pose = np.degrees(data[f]).reshape(8, 3)  # turn 24 numbers into 8 legs
        viz._update_leg_lines(lines, pose)        # put the legs in that pose
        ax.set_title(f"{name} - frame {f}")       # add a title
plt.tight_layout()
plt.savefig("predicted_poses.png", dpi=150)   # save the picture


plt.show()

# ===== STEP 8: ARCHITECTURE COMPARISON =====
# Train several network shapes with the same data, learning rate and
# epochs, so the only difference is the architecture. Then compare.
import matplotlib.pyplot as plt

def train(sizes, lr=0.1, epochs=3000, seed=42):
    """Train a fresh network of the given shape; return its loss history."""
    global rng
    rng = np.random.default_rng(seed)        # same starting randomness for each
    p = init_weights(sizes)
    history = []
    for epoch in range(epochs):
        pred, acts = forward_with_cache(X, p)
        history.append(mse_loss(pred, Y))
        grads = backward(pred, Y, acts, p)
        p = [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(p, grads)]
    return history, p

architectures = [
    [2, 4, 24],               # tiny: 1 hidden layer, 4 neurons
    [2, 16, 24],              # small: 1 hidden layer
    [2, 64, 24],              # medium: 1 hidden layer
    [2, 64, 64, 24],          # chosen: 2 hidden layers
    [2, 128, 128, 128, 24],   # large: 3 hidden layers
]

plt.figure(figsize=(8, 5))
print(f"{'Architecture':<28}{'Params':>8}{'Final loss':>14}")
for sizes in architectures:
    history, p = train(sizes)
    n_params = sum(W.size + b.size for W, b in p)
    print(f"{str(sizes):<28}{n_params:>8}{history[-1]:>14.6f}")
    plt.plot(history, label=f"{sizes}  ({n_params} params)")

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE loss (log scale)")
plt.title("Architecture comparison (lr = 0.1, 3000 epochs)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("architecture_comparison.png", dpi=150)
plt.show()