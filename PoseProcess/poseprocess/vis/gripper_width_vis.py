from typing import Iterable, Optional, Sequence, Union

import matplotlib.pyplot as plt


def plot_gripper_widths(
    widths: Union[Sequence[float], Iterable[float]],
    *,
    title: str = "Gripper Width over Index",
    x_label: str = "Index",
    y_label: str = "Gripper Width",
    grid: bool = True,
    figsize: tuple = (8, 4),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot gripper widths against their indices.

    Args:
        widths: A sequence/iterable of gripper widths. The x-axis will be the index.
        title: Figure title.
        x_label: Label for x axis (index).
        y_label: Label for y axis (gripper width).
        grid: Whether to show grid.
        figsize: Matplotlib figure size.
        save_path: If provided, saves the figure to this path.
        show: Whether to display the figure with plt.show(). If False and save_path is None,
              the figure will be closed after creation.
    """

    y = list(widths)
    x = list(range(len(y)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, marker="o", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if grid:
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


__all__ = ["plot_gripper_widths"]


