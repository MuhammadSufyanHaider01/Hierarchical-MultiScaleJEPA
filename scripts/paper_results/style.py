"""IEEE-oriented settings shared by paper result figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 7.16
COLORS = {"p0": "#0072B2", "p1": "#D55E00", "p2": "#009E73", "p2_reverse": "#CC79A7"}
METHODS = {"p0": "LatentWave", "p1": "Same-scale", "p2": "Fine-to-coarse", "p2_reverse": "Coarse-to-fine"}
TASKS = {"rfs": "RF classification", "jamming": "Jamming", "human-activity": "Human activity", "interference": "Interference", "nrpos-outdoor": "NR positioning", "deepmimo-beam": "Beam prediction"}
TASK_ORDER = tuple(TASKS)

def configure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.7, "lines.linewidth": 1.2, "lines.markersize": 4,
        "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
        "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white"})

def save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    plt.close(fig)
