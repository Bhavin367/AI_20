# Load the saved network and test it
import numpy as np

data = np.load("trained_model.npz")   # open the saved file
params = [(data["W0"], data["b0"]), (data["W1"], data["b1"]), (data["W2"], data["b2"])]   # get the 3 layers back

def forward(X, params):          # same maths as the network
    a = X
    for W, b in params[:-1]:     # middle layers
        a = np.tanh(a @ W + b)
    W, b = params[-1]            # last layer
    return a @ W + b

frame = 150                                      # pick a frame (0 to 299)
phase = 2 * np.pi * frame / 300                  # where we are in the walk
x = np.array([[np.sin(phase), np.cos(phase)]])   # the 2 inputs
angles = forward(x, params)                      # ask the network
print(f"Frame {frame} angles (degrees):")
print(np.degrees(angles).round(1))