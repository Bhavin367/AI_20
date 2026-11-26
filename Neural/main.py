"""Part 2 entry point: train the spider gait network on the evolved gait.

    python main.py              # full run, reads gait.csv
    python main.py --quick      # short run to check everything works

Steps: load gait.csv -> gradient check -> experiments -> final Adam training
-> save model + predicted gait -> evaluation -> PyTorch comparison (if installed).
Outputs (plots, spider_nn.npz, gait_predicted.csv) go to results/.
"""
import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import nntrainingmethod as nnt
import nn_backprop_check as bc
import compare_pytorch as cp

OUT = "results"
LEGS = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
JOINT_LABELS = [f"{leg} {j}" for leg in LEGS for j in ["coxa", "femur", "tibia"]]


def plot_all_joints(Y, pred, show):
    """8 legs x 3 joints: target (thick) vs network (dashed)."""
    fig, axes = plt.subplots(8, 3, figsize=(12, 16), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(np.rad2deg(Y[:, j]), linewidth=3, alpha=0.35)
        ax.plot(np.rad2deg(pred[:, j]), "--", color="tab:red", linewidth=1)
        ax.set_title(JOINT_LABELS[j], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    fig.supxlabel("Frame in walk cycle")
    fig.supylabel("Angle (degrees)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "all_joints.png"), dpi=130)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gait", default="gait.csv")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--quick", action="store_true", help="short run for testing")
    parser.add_argument("--no-show", action="store_true", help="save plots, don't open windows")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    show = not args.no_show
    epochs = 500 if args.quick else args.epochs
    os.makedirs(OUT, exist_ok=True)

    # data + backprop check
    X, Y = bc.load_data(args.gait)
    bc.gradient_check(X, Y)

    # experiments + final training
    print("=== Experiments ===")
    nnt.run_experiments(X, Y, epochs, OUT, show, args.quick)
    print("\n=== Architecture comparison (GD, lr = 0.1) ===")
    nnt.compare_architectures(X, Y, epochs, OUT, show, args.quick)
    print("\n=== Final training ===")
    params, _ = bc.run_convergence(X, Y, epochs, OUT, show)

    # save model + export gait (radians, 300 x 24, for the MATLAB spider)
    nnt.save_model(params, os.path.join(OUT, "spider_nn.npz"))
    pred = nnt.predict(params, len(X))
    np.savetxt(os.path.join(OUT, "gait_predicted.csv"), pred, delimiter=",")

    # evaluation
    err = pred - Y
    per_joint = np.sqrt(np.mean(err ** 2, axis=0))
    worst = int(np.argmax(per_joint))
    print("\n=== Evaluation ===")
    print(f"  RMSE {np.rad2deg(np.sqrt(np.mean(err ** 2))):.3f} deg, "
          f"max error {np.rad2deg(np.abs(err).max()):.3f} deg, "
          f"worst joint {JOINT_LABELS[worst]} ({np.rad2deg(per_joint[worst]):.3f} deg)")
    plot_all_joints(Y, pred, show)

    print("\n=== Comparison with PyTorch ===")
    cp.compare(X, Y, epochs, 0.001, out_dir=OUT, show=show)
    print(f"Done. Outputs are in {OUT}/")


if __name__ == "__main__":
    main()
