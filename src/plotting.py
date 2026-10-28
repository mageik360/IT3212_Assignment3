from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_figure(fig, path, dpi: int = 120) -> Path:
    """Save a figure (creating its folder), close it and return the file path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Figures created with a layout engine (e.g. constrained) handle spacing themselves
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
