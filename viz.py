"""Visualisation tools for the spider GA (matplotlib only, no scipy).

Every plot function saves a PNG (or GIF) and does NOT call plt.show(), so
main.py can show everything once at the end with plt.show().
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import config
import chrom_encode as ce
import fitness as fit

SEGS = config.segment_length
GROUND = 3.0                                   # half-size of the drawn floor


# ---------------------------------------------------------------------------
# 1. Fitness progression. Fitness = 1/(1+cost) is squashed near 0, so we plot
#    the total cost on a log axis, which shows early progress much better.
# ---------------------------------------------------------------------------
def plot_histories(runs, path="fitness.png"):
    """runs: {label: (best_hist, mean_hist)}"""
    plt.figure(figsize=(7, 4))
    for label, (best, mean) in runs.items():
        line, = plt.semilogy(1 / np.array(best) - 1, label=f"{label} best")
        plt.semilogy(1 / np.array(mean) - 1, "--", color=line.get_color(),
                     alpha=0.5, label=f"{label} mean")
    plt.xlabel("generation")
    plt.ylabel("total cost (log scale, lower is better)")
    plt.title("GA progress")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)


# ---------------------------------------------------------------------------
# 2. Cost breakdown per fitness term (weighted, as used in total_cost)
# ---------------------------------------------------------------------------
TERMS = {
    "limit": (fit.limit_cost, config.w_limit),
    "sym": (fit.sym_cost, config.w_sym),
    "ground": (fit.ground_cost, config.w_ground),
    "lift": (fit.lift_cost, config.w_lift),
    "stride": (fit.stride_cost, config.w_stride),
    "slip": (fit.slip_cost, config.w_slip),
}


def plot_costs(genomes, path="costs.png"):
    """genomes: {label: 96-gene genome}"""
    names = list(TERMS)
    x = np.arange(len(names))
    w = 0.8 / len(genomes)
    plt.figure(figsize=(8, 4))
    for i, (label, g) in enumerate(genomes.items()):
        vals = [fn(g) * wt + 1 for fn, wt in TERMS.values()]   # +1: zeros show on log axis
        plt.bar(x + i * w, vals, w, label=label)
    plt.xticks(x + w * (len(genomes) - 1) / 2, names)
    plt.yscale("log")
    plt.ylabel("weighted cost + 1 (log)")
    plt.title("Cost breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)


# ---------------------------------------------------------------------------
# 3. Drawing the spider. Same maths as fit.foot_positions but keeps every
#    joint so we can draw the whole leg. pose: (8, 3) in degrees.
# ---------------------------------------------------------------------------
def leg_points(pose):
    q = np.deg2rad(pose)
    e = config.coxa_elevation
    legs = []
    for i in range(config.n_legs):
        heading = fit.BASE[i] + q[i, 0]
        tilts = [e, e + q[i, 1], e + q[i, 1] + q[i, 2]]
        p = np.array([fit.MOUNT_X[i], fit.MOUNT_Y[i], 0.0])
        pts = [p.copy()]
        for length, t in zip(SEGS, tilts):
            p = p + [length * np.cos(t) * np.cos(heading),
                     length * np.cos(t) * np.sin(heading),
                     length * np.sin(t)]
            pts.append(p.copy())
        legs.append(np.array(pts))
    return legs


def _setup_axes(ax):
    th = np.linspace(0, 2 * np.pi, 60)
    ax.plot(config.A * np.cos(th), config.B * np.sin(th), 0, "k")        # body outline
    g = GROUND
    ax.plot_surface(*np.meshgrid([-g, g], [-g, g]),
                    np.full((2, 2), config.ground_z), alpha=0.15, color="gray")
    ax.set_xlim(-g, g)
    ax.set_ylim(-g, g)
    ax.set_zlim(-1.5, 1.5)
    ax.set_box_aspect((1, 1, 0.5))


def _make_leg_lines(ax):
    # left legs blue, right legs red (order in the genome: L1-L4 then R4-R1)
    return [ax.plot([], [], [], "-o", ms=3, color="tab:blue" if i < 4 else "tab:red")[0]
            for i in range(config.n_legs)]


def _update_leg_lines(lines, pose):
    for ln, pts in zip(lines, leg_points(pose)):
        ln.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])


def plot_key_frames(genome, path="poses.png"):
    frames = ce.to_frames(genome)
    fig = plt.figure(figsize=(3.2 * ce.N_FRAMES, 3.4))
    for t in range(ce.N_FRAMES):
        ax = fig.add_subplot(1, ce.N_FRAMES, t + 1, projection="3d")
        _setup_axes(ax)
        lines = _make_leg_lines(ax)
        _update_leg_lines(lines, frames[t])
        swing = [fit.LEG_NAMES[i] for i in np.where(fit.SWING[t])[0]]
        ax.set_title(f"frame {t}\nswing: {', '.join(swing) or '-'}", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)


# ---------------------------------------------------------------------------
# 4. Full gait: 4 key frames -> ~300 poses (the brief's "about 300 1x24
#    vectors"). Linear, periodic interpolation (frame 3 wraps back to 0).
#    NOTE: fitness only judges the 4 key frames, so in-between poses are an
#    approximation.
# ---------------------------------------------------------------------------
def full_gait(genome, steps=300):
    key = ce.to_frames(genome).reshape(ce.N_FRAMES, -1)           # (4, 24)
    closed = np.vstack([key, key[:1]])
    xs = np.arange(ce.N_FRAMES + 1)
    t = np.linspace(0, ce.N_FRAMES, steps, endpoint=False)
    return np.stack([np.interp(t, xs, closed[:, j]) for j in range(key.shape[1])], axis=1)


def animate(genome, path="walk.gif", steps=300):
    gait = full_gait(genome, steps)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    _setup_axes(ax)
    lines = _make_leg_lines(ax)

    def update(i):
        _update_leg_lines(lines, gait[i].reshape(config.n_legs, 3))
        ax.set_title(f"step {i + 1}/{steps}")
        return lines

    ani = FuncAnimation(fig, update, frames=steps, interval=40)
    ani.save(path, writer=PillowWriter(fps=25))
    plt.close(fig)
    return gait


def export_gait(genome, path="gait.csv", steps=300):
    """(steps x 24) matrix in DEGREES, leg order L1 L2 L3 L4 R4 R3 R2 R1.
    Check units/order against the MATLAB spider model before using it there."""
    gait = full_gait(genome, steps)
    np.savetxt(path, gait, delimiter=",")
    return gait
