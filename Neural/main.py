"""
main.py - runs the whole neural network solution.

    python main.py                      everything
    python main.py --skip-experiments   check, train, save, evaluate, plot
    python main.py --test               load the saved model and test it only

Needs numpy and matplotlib, with gait.csv in the same folder.
Outputs go to the results/ folder.
"""
import argparse

import matplotlib
matplotlib.use("Agg")                      # save figures without opening windows

import numpy as np

import neuralnetwork as nn
import nn_backprop_check as check
import nntrainingmethod as training

RESULTS = nn.HERE / "results"
MODEL_FILE = RESULTS / "spider_model.npz"

# ===== FINAL CONFIGURATION ==================================================
# Check these against results/experiments.csv. If another setting does clearly
# better on the gait, change it here, run again and explain why.
FINAL = {
    "layers": [2, 64, 64, nn.N_JOINTS],    # input (sin, cos) -> 24 joint angles
    "activation": "tanh",
    "optimiser": "adam",
    "lr": 0.001,
    "batch_size": None,                    # None = full batch
    "epochs": 3000,
    "seed": 42,
}


def heading(text):
    print(f"\n===== {text} =====")


def evaluate_and_plot(params, activation, keyframes):
    """Print the test results, save them, and draw the output plots."""
    report = "\n".join(check.evaluate(params, activation, keyframes))
    print(report)
    (RESULTS / "evaluation.txt").write_text(report + "\n")

    X, Y, phase = nn.make_dataset(keyframes, offset=0.5)
    pred = nn.predict(X, params, activation)
    check.plot_predicted_vs_target(pred, Y, phase, len(keyframes),
                                   RESULTS / "predicted_vs_target.png")
    check.plot_joint_errors(pred, Y, RESULTS / "joint_errors.png")
    check.plot_poses(params, activation, keyframes, RESULTS / "poses.png")


def main():
    parser = argparse.ArgumentParser(description="Spider gait neural network")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-experiments", action="store_true",
                      help="do not run the comparison experiments")
    mode.add_argument("--test", action="store_true",
                      help="test the saved model without training")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    heading("Data")
    keyframes = nn.load_keyframes()
    X, Y, _ = nn.make_dataset(keyframes)
    print(f"  key frames {keyframes.shape}, angles {keyframes.min():.1f} to "
          f"{keyframes.max():.1f} deg")
    print(f"  training inputs {X.shape}, targets {Y.shape} (radians)")

    if args.test:
        heading("Testing the saved model")
        params, activation = nn.load_model(MODEL_FILE)
        evaluate_and_plot(params, activation, keyframes)
        print(f"\nPlots saved in {RESULTS}")
        return

    heading("1. Gradient check (backprop vs numerical gradient)")
    for activation in nn.ACTIVATIONS:
        print(f"\n  activation: {activation}")
        worst = check.gradient_check(X, Y, FINAL["layers"], activation)
        if worst > 1e-4:
            raise SystemExit(f"Gradient check FAILED ({worst:.2e}). Fix backward().")
        print(f"  PASS (worst {worst:.2e})")

    if not args.skip_experiments:
        heading("2. Experiments (one setting changed at a time)")
        training.run_experiments(X, Y, FINAL, RESULTS)

    heading("3. Training the final network")
    for key, value in FINAL.items():
        print(f"  {key:<12}{value}")
    history, params = training.train(X, Y, **FINAL)
    print(f"  parameters  {nn.count_parameters(params)}")
    for epoch in range(0, len(history), max(1, FINAL["epochs"] // 6)):
        print(f"  epoch {epoch:>6}   loss {history[epoch]:.7f}")
    print(f"  final loss {history[-1]:.7f}")
    check.plot_convergence(
        history, f"Training the final network ({FINAL['optimiser']}, lr = {FINAL['lr']})",
        RESULTS / "convergence.png")

    heading("4. Saving the model")
    nn.save_model(MODEL_FILE, params, FINAL["activation"])
    loaded, activation = nn.load_model(MODEL_FILE)
    if not np.array_equal(nn.predict(X, loaded, activation),
                          nn.predict(X, params, FINAL["activation"])):
        raise SystemExit("The reloaded model does not match the trained one.")
    print(f"  saved and reloaded correctly: {MODEL_FILE}")

    heading("5. Evaluation")
    evaluate_and_plot(loaded, activation, keyframes)
    print(f"\nDone. Results are in {RESULTS}")


if __name__ == "__main__":
    main()
