"""
Побудова графіків: рівняння системи, цільова функція, траєкторії спуску.
Лабораторна робота №9
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os


def plot_system_equations(exact_solutions, save_path="results/system_plot.png"):
    """
    Крок 1: Будує графіки рівнянь системи нелінійних рівнянь.
      f1: x1^2 + x2^2 = 4  (коло)
      f2: x1 * x2 = 1      (гіпербола)
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("#0f1117")
    fig.patch.set_facecolor("#0f1117")

    x = np.linspace(-3, 3, 800)

    # Коло
    mask = 4 - x**2 >= 0
    y_upper =  np.sqrt(np.where(mask, 4 - x**2, np.nan))
    y_lower = -np.sqrt(np.where(mask, 4 - x**2, np.nan))
    ax.plot(x, y_upper, color="#00d4ff", lw=2.5, label=r"$x_1^2 + x_2^2 = 4$")
    ax.plot(x, y_lower, color="#00d4ff", lw=2.5)

    # Гіпербола
    xp = np.linspace(0.25, 3, 500)
    xn = np.linspace(-3, -0.25, 500)
    ax.plot(xp,  1.0/xp, color="#ff6b35", lw=2.5, label=r"$x_1 \cdot x_2 = 1$")
    ax.plot(xn,  1.0/xn, color="#ff6b35", lw=2.5)

    # Точки перетину (розв'язки)
    for s in exact_solutions:
        ax.scatter(s[0], s[1], s=130, color="#ffe600", zorder=5,
                   edgecolors="white", lw=1.5)
        ax.annotate(f"({s[0]:.3f}, {s[1]:.3f})",
                    xy=(s[0], s[1]),
                    xytext=(s[0]+0.12, s[1]+0.18),
                    color="white", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#0f1117", alpha=0.7))

    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title("Система нелінійних рівнянь (4 розв'язки)", color="white", fontsize=13, pad=14)
    ax.set_xlabel("$x_1$", color="white", fontsize=12)
    ax.set_ylabel("$x_2$", color="white", fontsize=12)
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
    ax.legend(facecolor="#1c1f26", edgecolor="gray", labelcolor="white", fontsize=11)
    ax.grid(True, color="gray", alpha=0.15)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Графік системи збережено: {save_path}")


def plot_objective_and_trajectory(trajectory, f_func,
                                   title="Цільова функція та траєкторія спуску",
                                   save_path="results/objective_trajectory.png"):
    """Контурний графік цільової функції та траєкторія спуску."""
    pts = np.array(trajectory)
    cx, cy = pts[:, 0], pts[:, 1]

    margin = max(1.0, 0.5 * (max(cx)-min(cx) + max(cy)-min(cy)))
    x_lo = min(cx) - margin
    x_hi = max(cx) + margin
    y_lo = min(cy) - margin
    y_hi = max(cy) + margin

    X, Y = np.meshgrid(np.linspace(x_lo, x_hi, 300),
                       np.linspace(y_lo, y_hi, 300))
    Z = np.vectorize(lambda a, b: f_func(np.array([a, b])))(X, Y)
    Z = np.clip(Z, 1e-12, None)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor("#0f1117")
    fig.patch.set_facecolor("#0f1117")

    cf = ax.contourf(X, Y, Z, levels=40, cmap="inferno", norm=LogNorm())
    cbar = fig.colorbar(cf, ax=ax)
    cbar.ax.tick_params(colors="white")
    ax.contour(X, Y, Z, levels=20, colors="white", alpha=0.2, linewidths=0.5)

    ax.plot(cx, cy, "o-", color="#00ffcc", lw=1.8, ms=4, alpha=0.85, label="Траєкторія")
    ax.scatter(cx[0],  cy[0],  s=120, color="#ffe600", zorder=6, label="Старт")
    ax.scatter(cx[-1], cy[-1], s=150, color="#00ffcc", zorder=6,
               marker="*", edgecolors="white", lw=1, label="Мінімум")

    ax.set_title(title, color="white", fontsize=12, pad=12)
    ax.set_xlabel("$x_1$", color="white", fontsize=12)
    ax.set_ylabel("$x_2$", color="white", fontsize=12)
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
    ax.legend(facecolor="#1c1f26", edgecolor="gray", labelcolor="white", fontsize=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Графік траєкторії збережено: {save_path}")


def plot_convergence(trajectories_dict, f_funcs_dict,
                     save_path="results/convergence.png"):
    """Порівняльний графік збіжності для кількох функцій."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#0f1117")
    fig.patch.set_facecolor("#0f1117")

    colors = ["#00d4ff", "#ff6b35", "#ffe600", "#00ffcc", "#ff4081", "#b388ff"]
    for (name, traj), color in zip(trajectories_dict.items(), colors):
        f = f_funcs_dict[name]
        vals = [f(pt) for pt in traj]
        if any(v > 0 for v in vals):
            ax.semilogy(vals, lw=2, color=color, label=name)

    ax.set_title("Збіжність методу Хука-Дживса", color="white", fontsize=13, pad=12)
    ax.set_xlabel("Крок", color="white", fontsize=11)
    ax.set_ylabel("f(x) (log-шкала)", color="white", fontsize=11)
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
    ax.legend(facecolor="#1c1f26", edgecolor="gray", labelcolor="white", fontsize=9)
    ax.grid(True, color="gray", alpha=0.2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Графік збіжності збережено: {save_path}")