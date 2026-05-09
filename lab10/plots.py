# plots.py
# ============================================================
# Побудова ВСІХ графіків лабораторної роботи №10
# (Частина 1 — Адамс, Частина 2 — Рунге-Кутта 4-го порядку)
#
# Кожен графік відповідає конкретному кроку хід роботи
# з методички. Графіки зберігаються як окремі PNG-файли
# та виводяться на екран (якщо show=True).
#
# Запуск:  python plots.py
# ============================================================

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, LogFormatter
from matplotlib.patches import FancyArrowPatch

# --- Додаємо поточну директорію до пошукового шляху ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exact_solution import f, y_exact, X0, Y0, X_END, H, EPS
from part1_adams import (
    solve_adams2, rk4_step as rk4_step_a,
    local_error_exact as err_exact_a,
    runge_error_estimate_adams2,
    solve_adams2_adaptive,
)
from part2_runge_kutta import (
    rk4_step, solve_rk4,
    local_error_exact as err_exact_rk,
    study_error_vs_h,
    runge_error_estimate_rk4,
    solve_rk4_adaptive,
)

# ============================================================
# Налаштування стилю
# ============================================================
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'axes.titleweight':  'bold',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.35,
    'grid.linestyle':    '--',
    'lines.linewidth':   2.0,
    'legend.fontsize':   9,
    'legend.framealpha': 0.85,
    'figure.dpi':        130,
    'savefig.dpi':       150,
    'savefig.bbox':      'tight',
})

# Кольорова палітра
C_EXACT  = '#1565C0'   # темно-синій  — точний розв'язок
C_ADAMS  = '#E53935'   # червоний     — Адамс
C_RK4    = '#2E7D32'   # темно-зелений — РК-4
C_PRED   = '#FB8C00'   # помаранчевий — прогноз
C_ERR    = '#6A1B9A'   # фіолетовий   — похибка
C_RUNGE  = '#00838F'   # бірюзовий    — Рунге-оцінка
C_ADAPT  = '#C62828'   # темно-червоний — адаптивний крок
C_EPS    = '#B71C1C'   # межа точності
C_HOPT   = '#1B5E20'   # h_opt лінія
C_STEP   = '#0D47A1'   # крок сітки


# ============================================================
# УТИЛІТИ
# ============================================================
def _add_formula_box(ax, text, x=0.03, y=0.97):
    """Додає блок з формулою у лівий верхній кут графіка."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=8.5, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4',
                      edgecolor='#F9A825', alpha=0.9))


def _save(fig, name):
    fig.savefig(name)
    print(f"  ✔ Збережено: {name}")


# ============================================================
# КРОК 1  —  Аналітичний розв'язок
# ============================================================
def plot_step1_exact(show=True):
    """
    Крок 1: Графік точного аналітичного розв'язку задачі Коші.
    Рівняння: y' = -2y + x,  y(0)=1,  x∈[0,1]
    Точний розв'язок: y(x) = x/2 - 1/4 + 5/4·exp(-2x)
    """
    xs  = np.linspace(X0, X_END, 600)
    ys  = y_exact(xs)
    dys = f(xs, ys)   # y'(x) = f(x, y(x))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Крок 1 — Аналітичний розв'язок задачі Коші\n"
                 r"$y' = -2y + x,\quad y(0)=1,\quad x\in[0,1]$", fontsize=13)

    # --- Ліворуч: y(x) ---
    ax = axes[0]
    ax.plot(xs, ys, color=C_EXACT, lw=2.5, label=r'$y(x)=\frac{x}{2}-\frac{1}{4}+\frac{5}{4}e^{-2x}$')
    ax.scatter([X0], [Y0], color='red', zorder=5, s=70, label=f'Початкова умова $y({X0})={Y0}$')
    ax.set_xlabel("x"); ax.set_ylabel("y(x)")
    ax.set_title("Точний розв'язок y(x)")
    ax.legend(loc='upper right')
    _add_formula_box(ax,
        "y(x) = x/2 - 1/4\n"
        "      + (5/4)·exp(-2x)")

    # --- Праворуч: y'(x) = f(x, y) ---
    ax = axes[1]
    ax.plot(xs, dys, color=C_PRED, lw=2.5, label=r"$y'(x) = f(x,y) = -2y+x$")
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel("x"); ax.set_ylabel("y'(x)")
    ax.set_title("Похідна y'(x) = f(x, y(x))")
    ax.legend()
    _add_formula_box(ax, "f(x, y) = -2·y + x")

    plt.tight_layout()
    _save(fig, "plot_step1_exact.png")
    if show: plt.show()
    plt.close(fig)


# ============================================================
# КРОК 2  —  Розв'язок методом Адамса 2-го порядку
# ============================================================
def plot_step2_adams_solution(show=True):
    """
    Крок 2: Чисельний розв'язок методом прогнозу-корекції Адамса 2-го порядку.
    Порівняння: точний vs прогноз vs скорегований.

    Формули прогнозу/корекції Адамса 2-го порядку:
      Прогноз:  y*_{n+1} = y_n + h/2·(3f_n - f_{n-1})
      Корекція: y^{k+1}_{n+1} = y_n + h/2·(f(x_{n+1}, y^k_{n+1}) + f_n)
    """
    xs_ex = np.linspace(X0, X_END, 600)
    ys_ex = y_exact(xs_ex)

    xs, ys, y_preds = solve_adams2(X0, Y0, X_END, H)
    ys_true = y_exact(xs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Крок 2 — Метод прогнозу та корекції Адамса 2-го порядку\n"
                 r"$y^*_{n+1}=y_n+\frac{h}{2}(3f_n-f_{n-1})$"
                 r"$\quad|\quad$"
                 r"$y^{(k+1)}_{n+1}=y_n+\frac{h}{2}(f_{n+1}^{(k)}+f_n)$", fontsize=12)

    # --- Ліворуч: порівняння кривих ---
    ax = axes[0]
    ax.plot(xs_ex, ys_ex, color=C_EXACT, lw=2.5, zorder=3,
            label='Точний розв\'язок')
    ax.plot(xs, y_preds, color=C_PRED, lw=1.5, ls='--', marker='^',
            markersize=6, zorder=2, label='Прогноз $y^*_{n+1}$')
    ax.plot(xs, ys, color=C_ADAMS, lw=1.8, ls='-.', marker='o',
            markersize=5, zorder=4, label='Після корекції $y_{n+1}$')
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Розв'язок на сітці, h = {H}")
    ax.legend()
    _add_formula_box(ax,
        "Прогноз:\n"
        "  y* = y_n + h/2·(3f_n - f_{n-1})\n"
        "Корекція (ітерації):\n"
        "  y^{k+1} = y_n + h/2·(f*_{n+1}+f_n)")

    # --- Праворуч: різниця прогноз vs скорегований ---
    ax = axes[1]
    diff = np.abs(y_preds - ys)
    ax.semilogy(xs, diff + 1e-16, color=C_PRED, lw=2, marker='^', ms=6,
                label=r'$|y^*_{n+1} - y_{n+1}|$ (прогноз - корекція)')
    ax.semilogy(xs, np.abs(ys_true - ys) + 1e-16, color=C_ADAMS,
                lw=2, marker='o', ms=5, label=r'$|y_{exact} - y_{Adams}|$')
    ax.set_xlabel("x"); ax.set_ylabel("Різниця (лог. масштаб)")
    ax.set_title("Вплив корекції на точність")
    ax.legend()

    plt.tight_layout()
    _save(fig, "plot_step2_adams_solution.png")
    if show: plt.show()
    plt.close(fig)


# ============================================================
# КРОК 3  —  Локальна похибка Адамса (через точний розв'язок)
# ============================================================
def plot_step3_adams_local_error(show=True):
    """
    Крок 3: Графік локальної похибки методу Адамса
    δ_n = |y_exact(x_n) - y_Adams(x_n)|
    для різних значень кроку h.
    """
    h_list  = [0.2, 0.1, 0.05, 0.025]
    colors  = [C_EXACT, C_ADAMS, C_RK4, C_PRED]
    markers = ['o', 's', '^', 'D']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"Крок 3 — Локальна похибка Адамса 2-го порядку"
                 "\n"
                 r"$\delta_n = |y_{exact}(x_n) - y_{Adams}(x_n)|$", fontsize=13)

    # --- Ліворуч: похибка в лінійному масштабі ---
    ax = axes[0]
    for h_i, col, mk in zip(h_list, colors, markers):
        xs, ys, _ = solve_adams2(X0, Y0, X_END, h_i)
        err = np.abs(y_exact(xs) - ys)
        ax.plot(xs, err, color=col, lw=1.8, marker=mk, ms=5,
                label=f'h = {h_i}')
    ax.set_xlabel("x"); ax.set_ylabel(r'$\delta_n$')
    ax.set_title("Похибка (лінійний масштаб)")
    ax.legend()
    _add_formula_box(ax, r"δ_n = |y_exact - y_Adams|")

    # --- Праворуч: похибка в логарифмічному масштабі ---
    ax = axes[1]
    for h_i, col, mk in zip(h_list, colors, markers):
        xs, ys, _ = solve_adams2(X0, Y0, X_END, h_i)
        err = np.abs(y_exact(xs) - ys)
        ax.semilogy(xs, err + 1e-16, color=col, lw=1.8, marker=mk, ms=5,
                    label=f'h = {h_i}')
    ax.set_xlabel("x"); ax.set_ylabel(r'$\delta_n$ (лог.)')
    ax.set_title("Похибка (логарифмічний масштаб)")
    ax.legend()

    plt.tight_layout()
    _save(fig, "plot_step3_adams_error.png")
    if show: plt.show()
    plt.close(fig)


# ============================================================
# КРОК 4  —  Оцінка похибки Адамса методом Рунге
# ============================================================
def plot_step4_adams_runge(show=True):
    """
    Крок 4: Оцінка похибки методом Рунге та перевірка оптимальності кроку.

    Формула оцінки (s=2, метод 2-го порядку):
      ε_n = |y_{n+1}(h) - y_{n+1}(h/2)| / (2^2 - 1)
          = |y_{n+1}(h) - y_{n+1}(h/2)| / 3
    """
    s = 2
    divisor = 2**s - 1   # = 3

    h_list = [0.2, 0.1, 0.05]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Крок 4 — Оцінка похибки Адамса методом Рунге\n"
                 r"$\varepsilon_n = \frac{|y(h) - y(h/2)|}{2^s-1},\quad s=2$", fontsize=13)

    # --- Ліворуч: Рунге-оцінка при різних h ---
    ax = axes[0]
    colors  = [C_EXACT, C_RUNGE, C_ADAMS]
    markers = ['o', 's', '^']
    for h_i, col, mk in zip(h_list, colors, markers):
        xs_r, eps_r = runge_error_estimate_adams2(X0, Y0, X_END, h_i, s=s)
        ax.semilogy(xs_r, eps_r + 1e-16, color=col, lw=1.8,
                    marker=mk, ms=5, label=f'h = {h_i}')
    ax.axhline(EPS, color=C_EPS, lw=1.8, ls='--',
               label=f'Задана точність ε = {EPS}')
    ax.set_xlabel("x"); ax.set_ylabel(r'$\varepsilon_n$ (лог.)')
    ax.set_title("Рунге-оцінка при різних h")
    ax.legend()
    _add_formula_box(ax, f"ε_n = |y(h)-y(h/2)| / {divisor}\n(s={s}, Адамс 2-го пор.)")

    # --- Праворуч: порівняння Рунге-оцінки та точної похибки ---
    ax = axes[1]
    xs_r, eps_r = runge_error_estimate_adams2(X0, Y0, X_END, H, s=s)
    xs_e, ys_e, _ = solve_adams2(X0, Y0, X_END, H)
    err_true = np.abs(y_exact(xs_e) - ys_e)
    ax.semilogy(xs_r, eps_r + 1e-16, color=C_RUNGE, lw=2, marker='s', ms=6,
                label=r'Рунге-оцінка $\varepsilon_n$')
    ax.semilogy(xs_e, err_true + 1e-16, color=C_ADAMS, lw=2, ls='--',
                marker='o', ms=5, label=r'Точна похибка $\delta_n$')
    ax.axhline(EPS, color=C_EPS, lw=1.5, ls=':', label=f'EPS = {EPS}')

    # Позначаємо зони
    ax.axhspan(EPS / (2**s), EPS, alpha=0.08, color='green', label='Оптимальна зона')
    ax.axhspan(EPS, ax.get_ylim()[1] if ax.get_ylim()[1] > EPS else EPS*10,
               alpha=0.06, color='red')
    ax.set_xlabel("x"); ax.set_ylabel("Похибка (лог.)")
    ax.set_title(f"Рунге vs точна похибка (h={H})")
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "plot_step4_adams_runge.png")
    if show: plt.show()
    plt.close(fig)


# ============================================================
# КРОК 5  —  Адаптивний крок Адамса
# ============================================================
def plot_step5_adams_adaptive(show=True):
    """
    Крок 5: Автоматичний вибір кроку для методу Адамса.
    Правило (з методички):
      якщо ε > EPS        → h = h/2
      якщо ε ≤ EPS/C      → h = h·2  (C = 2^s = 4)
      інакше              → h без змін
    """
    xs_ad, ys_ad, hs_ad = solve_adams2_adaptive(X0, Y0, X_END, H, EPS)
    xs_ex = np.linspace(X0, X_END, 600)

    # Також фіксований крок для порівняння
    xs_fix, ys_fix, _ = solve_adams2(X0, Y0, X_END, H)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Крок 5 — Автоматичний вибір кроку (метод Адамса 2-го порядку)\n"
                 r"$\varepsilon > \mathrm{EPS} \Rightarrow h/2$;"
                 r"$\quad\varepsilon \leq \mathrm{EPS}/4 \Rightarrow 2h$", fontsize=12)

    # --- Ліворуч: розв'язок ---
    ax = axes[0]
    ax.plot(xs_ex, y_exact(xs_ex), color=C_EXACT, lw=2.5, label='Точний y(x)')
    ax.plot(xs_fix, ys_fix, color=C_ADAMS, ls='--', lw=1.5, marker='o',
            ms=5, label=f'Адамс фікс. h={H}')
    ax.plot(xs_ad, ys_ad, color=C_ADAPT, lw=1.8, ls='-.', marker='D',
            ms=6, label='Адамс адаптивний')
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Розв'язок")
    ax.legend(fontsize=8)

    # --- По центру: крок h(x) ---
    ax = axes[1]
    if len(hs_ad) > 0:
        ax.step(xs_ad[1:], hs_ad, color=C_STEP, lw=2.2, where='post',
                label='Адаптивний h(x)')
        ax.axhline(H, color='orange', lw=1.5, ls='--', label=f'Фікс. h={H}')
        ax.axhline(EPS, color=C_EPS, lw=1.2, ls=':', label=f'EPS={EPS}')
        ax.set_ylim(0, max(hs_ad) * 1.4)
    ax.set_xlabel("x"); ax.set_ylabel("h")
    ax.set_title("Залежність кроку h(x)")
    ax.legend(fontsize=8)
    _add_formula_box(ax,
        "ε > EPS  →  h = h/2\n"
        "ε ≤ EPS/4 →  h = h·2\n"
        "інакше   →  h = h")

    # --- Праворуч: порівняння похибок ---
    ax = axes[2]
    err_fix = np.abs(y_exact(xs_fix) - ys_fix)
    err_ad  = np.abs(y_exact(xs_ad)  - ys_ad)
    ax.semilogy(xs_fix, err_fix + 1e-16, color=C_ADAMS, lw=2, marker='o',
                ms=5, label=f'Фікс. h={H}')
    ax.semilogy(xs_ad, err_ad + 1e-16, color=C_ADAPT, lw=2, marker='D',
                ms=5, label='Адаптивний')
    ax.axhline(EPS, color=C_EPS, lw=1.5, ls='--', label=f'EPS={EPS}')
    ax.set_xlabel("x"); ax.set_ylabel("Похибка (лог.)")
    ax.set_title("Похибка: фікс. vs адаптивний")
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "plot_step5_adams_adaptive.png")
    if show: plt.show()
    plt.close(fig)


# ============================================================
# КРОК 6  —  Розв'язок методом Рунге-Кутта 4-го порядку
# ============================================================
def plot_step6_rk4_solution(show=True):
    """
    Крок 6: Чисельний розв'язок методом Рунге-Кутта 4-го порядку.
    Формули:
      k1 = f(x_n,       y_n)
      k2 = f(x_n+h/2,   y_n + h/2·k1)
      k3 = f(x_n+h/2,   y_n + h/2·k2)
      k4 = f(x_n+h,     y_n + h·k3)
      y_{n+1} = y_n + h/6·(k1 + 2k2 + 2k3 + k4)
    """
    xs_ex = np.linspace(X0, X_END, 600)
    ys_ex = y_exact(xs_ex)

    h_list  = [0.2, 0.1, 0.05]
    colors  = [C_ADAMS, C_RK4, C_PRED]
    markers = ['s', 'o', '^']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Крок 6 — Метод Рунге-Кутта 4-го порядку\n"
                 r"$y_{n+1}=y_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4)$", fontsize=13)

    # --- Ліворуч: розв'язки при різних h ---
    ax = axes[0]
    ax.plot(xs_ex, ys_ex, color=C_EXACT, lw=2.8, zorder=5,
            label='Точний y(x)')
    for h_i, col, mk in zip(h_list, colors, markers):
        xs, ys = solve_rk4(X0, Y0, X_END, h_i)
        ax.plot(xs, ys, color=col, lw=1.6, ls='--', marker=mk,
                ms=6, zorder=3, label=f'РК-4, h={h_i}')
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Розв'язок при різних h")
    ax.legend()
    _add_formula_box(ax,
        "k1 = f(x_n,     y_n)\n"
        "k2 = f(x+h/2,   y+h/2·k1)\n"
        "k3 = f(x+h/2,   y+h/2·k2)\n"
        "k4 = f(x+h,     y+h·k3)\n"
        "y_{n+1}=y_n+h/6·(k1+2k2+2k3+k4)")

    # --- Праворуч: коефіцієнти k1..k4 по вузлах ---
    xs, ys_rk = solve_rk4(X0, Y0, X_END, H)
    k1s, k2s, k3s, k4s = [], [], [], []
    x_, y_ = X0, Y0
    for xi, yi in zip(xs[:-1], ys_rk[:-1]):
        k1 = f(xi,        yi)
        k2 = f(xi + H/2,  yi + H/2*k1)
        k3 = f(xi + H/2,  yi + H/2*k2)
        k4 = f(xi + H,    yi + H*k3)
        k1s.append(k1); k2s.append(k2)
        k3s.append(k3); k4s.append(k4)

    ax = axes[1]
    xs_mid = xs[:-1]
    ax.plot(xs_mid, k1s, color='#1565C0', lw=1.8, marker='o', ms=5, label=r'$k_1$')
    ax.plot(xs_mid, k2s, color='#E53935', lw=1.8, marker='s', ms=5, label=r'$k_2$')
    ax.plot(xs_mid, k3s, color='#2E7D32', lw=1.8, marker='^', ms=5, label=r'$k_3$')
    ax.plot(xs_mid, k4s, color='#FB8C00', lw=1.8, marker='D', ms=5, label=r'$k_4$')
    ax.set_xlabel("x"); ax.set_ylabel("Значення коефіцієнта")
    ax.set_title(f"Коефіцієнти k₁…k₄ вздовж [0,1], h={H}")
    ax.legend()

    plt.tight_layout()
    _save(fig, "plot_step6_rk4_solution.png")
    if show: plt.show()
    plt.close(fig)


# ============================================================
# КРОК 7  —  Локальна похибка РК-4 та залежність від h
# ============================================================
def plot_step7_rk4_error(show=True):
    """
    Крок 7: Графіки локальної похибки РК-4 та дослідження
    залежності max|δ| від кроку h (очікуємо O(h^4)).

    δ_n = |y_exact(x_n) - y_rk4(x_n)|
    """
    h_list  = [0.2, 0.1, 0.05, 0.025, 0.01]
    hs_s, max_errs = study_error_vs_h(h_list)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(r"Крок 7 — Локальна похибка РК-4:  $\delta_n = |y_{exact}-y_{RK4}|$", fontsize=13)

    # --- Ліворуч: похибка вздовж x при різних h ---
    colors_h  = ['#1565C0', '#E53935', '#2E7D32', '#FB8C00', '#6A1B9A']
    markers_h = ['o', 's', '^', 'D', 'v']
    ax = axes[0]
    for h_i, col, mk in zip(h_list, colors_h, markers_h):
        xs, ys = solve_rk4(X0, Y0, X_END, h_i)
        err = np.abs(y_exact(xs) - ys)
        ax.semilogy(xs, err + 1e-16, color=col, lw=1.6, marker=mk,
                    ms=5, label=f'h={h_i}')
    ax.set_xlabel("x"); ax.set_ylabel(r'$\delta_n$ (лог.)')
    ax.set_title("Похибка вздовж x")
    ax.legend(fontsize=8)
    _add_formula_box(ax, "δ_n = |y_exact - y_RK4|")

    # --- По центру: max|δ|(h) у логлог масштабі ---
    ax = axes[1]
    ax.loglog(hs_s, max_errs, color=C_RK4, lw=2.2, marker='o', ms=8,
              label=r'$\max|\delta_n|(h)$')
    # Еталонна лінія O(h^4)
    h_ref = np.array(hs_s, dtype=float)
    ref4  = max_errs[0] * (h_ref / hs_s[0])**4
    ref2  = max_errs[0] * (h_ref / hs_s[0])**2
    ax.loglog(h_ref, ref4, color=C_PRED, lw=1.5, ls='--', label=r'$O(h^4)$ — еталон')
    ax.loglog(h_ref, ref2, color='gray', lw=1.2, ls=':', label=r'$O(h^2)$ — для порівн.')
    ax.set_xlabel("h (лог.)"); ax.set_ylabel(r'$\max|\delta_n|$ (лог.)')
    ax.set_title("Залежність max|δ| від h (логлог)")
    ax.legend()
    _add_formula_box(ax, "Очікуємо O(h^4)\nдля РК-4")

    # --- Праворуч: емпіричний порядок точності ---
    ax = axes[2]
    orders = []
    for i in range(1, len(hs_s)):
        if max_errs[i] > 0 and max_errs[i-1] > 0:
            p = (np.log(max_errs[i-1]) - np.log(max_errs[i])) / \
                (np.log(hs_s[i-1])    - np.log(hs_s[i]))
        else:
            p = 0
        orders.append(p)
    xs_ord = [(hs_s[i]+hs_s[i-1])/2 for i in range(1, len(hs_s))]
    ax.bar([str(round(h,3)) for h in xs_ord], orders, color=C_RUNGE,
           edgecolor='white', linewidth=0.5)
    ax.axhline(4, color=C_EPS, lw=1.8, ls='--', label='Теоретичний порядок p=4')
    ax.set_xlabel("h (середнє між сусідніми)"); ax.set_ylabel("Емпіричний порядок p")
    ax.set_title("Емпіричний порядок точності")
    ax.legend()
    ax.set_ylim(0, 6)
    _add_formula_box(ax,
        "p = log(E_{i-1}/E_i)\n"
        "  / log(h_{i-1}/h_i)")

    plt.tight_layout()
    _save(fig, "plot_step7_rk4_error.png")
    if show: plt.show()
    plt.close(fig)


# ============================================================
# КРОК 8  —  Оцінка похибки РК-4 методом Рунге
# ============================================================
def plot_step8_rk4_runge(show=True):
    """
    Крок 8: Оцінка похибки методом Рунге для РК-4 (s=4).
    ε_n = |y(h) - y(h/2)| / (2^4 - 1) = |y(h) - y(h/2)| / 15

    Також оцінюємо оптимальний крок:
      h_opt = h · (EPS / ε_max)^{1/s}
    """
    s = 4
    divisor = 2**s - 1  # = 15

    h_list = [0.2, 0.1, 0.05]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Крок 8 — Оцінка похибки РК-4 методом Рунге\n"
                 r"$\varepsilon_n = \frac{|y(h)-y(h/2)|}{2^4-1} = \frac{|y(h)-y(h/2)|}{15}$",
                 fontsize=13)

    # --- Ліворуч: Рунге-оцінка при різних h ---
    ax = axes[0]
    colors  = [C_EXACT, C_RK4, C_ADAMS]
    markers = ['o', 's', '^']
    for h_i, col, mk in zip(h_list, colors, markers):
        xs_r, eps_r, _ = runge_error_estimate_rk4(X0, Y0, X_END, h_i, s=s)
        ax.semilogy(xs_r, eps_r + 1e-16, color=col, lw=1.8, marker=mk,
                    ms=5, label=f'h={h_i}')
    ax.axhline(EPS, color=C_EPS, lw=1.8, ls='--', label=f'EPS={EPS}')
    ax.set_xlabel("x"); ax.set_ylabel(r'$\varepsilon_n$ (лог.)')
    ax.set_title("Рунге-оцінка при різних h")
    ax.legend()
    _add_formula_box(ax, f"ε_n = |y(h)-y(h/2)| / {divisor}")

    # --- По центру: Рунге vs точна похибка ---
    ax = axes[1]
    xs_r, eps_r, h_opt = runge_error_estimate_rk4(X0, Y0, X_END, H, s=s)
    xs_e, ys_e = solve_rk4(X0, Y0, X_END, H)
    err_true = np.abs(y_exact(xs_e) - ys_e)
    ax.semilogy(xs_r, eps_r + 1e-16, color=C_RUNGE, lw=2.2, marker='s',
                ms=6, label=r'Рунге $\varepsilon_n$')
    ax.semilogy(xs_e, err_true + 1e-16, color=C_RK4, lw=2, ls='--',
                marker='o', ms=5, label=r'Точна $\delta_n$')
    ax.axhline(EPS, color=C_EPS, lw=1.5, ls=':', label=f'EPS={EPS}')
    ax.set_xlabel("x"); ax.set_ylabel("Похибка (лог.)")
    ax.set_title(f"Рунге vs точна похибка (h={H})")
    ax.legend()

    # --- Праворуч: оцінка h_opt для різних EPS ---
    ax = axes[2]
    eps_targets = np.logspace(-2, -6, 50)
    _, eps_r_h, _ = runge_error_estimate_rk4(X0, Y0, X_END, H, s=s)
    eps_max_h = max(np.max(eps_r_h), 1e-16)
    h_opts = []
    for eps_t in eps_targets:
        if eps_max_h > 0:
            h_opt_i = H * (eps_t / eps_max_h) ** (1.0 / s)
        else:
            h_opt_i = H
        h_opts.append(h_opt_i)
    ax.loglog(eps_targets, h_opts, color=C_RK4, lw=2.5,
              label=r'$h_{opt}(\varepsilon)$')
    ax.axvline(EPS, color=C_EPS, lw=1.5, ls='--', label=f'Задана EPS={EPS}')
    ax.axhline(H, color='orange', lw=1.5, ls=':', label=f'Поточний h={H}')
    ax.set_xlabel(r'Задана точність $\varepsilon$')
    ax.set_ylabel(r'$h_{opt}$')
    ax.set_title("Оптимальний крок vs задана точність")
    ax.legend()
    _add_formula_box(ax, r"h_opt = h·(EPS/ε_max)^{1/s}")

    plt.tight_layout()
    _save(fig, "plot_step8_rk4_runge.png")
    if show: plt.show()
    plt.close(fig)


# ============================================================
# КРОК 9  —  Адаптивний крок РК-4
# ============================================================
def plot_step9_rk4_adaptive(show=True):
    """
    Крок 9: Автоматичний вибір кроку для РК-4.
    Правило (с=4, C=16):
      ε = |y(h) - y(h/2)| / 15
      ε > EPS         → h = h/2
      ε ≤ EPS/16      → h = h·2
      інакше          → h без змін
    """
    xs_ad, ys_ad, hs_ad = solve_rk4_adaptive(X0, Y0, X_END, H, EPS)
    xs_fix, ys_fix       = solve_rk4(X0, Y0, X_END, H)
    xs_ex                = np.linspace(X0, X_END, 600)

    _, eps_r, h_opt = runge_error_estimate_rk4(X0, Y0, X_END, H, s=4)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Крок 9 — Автоматичний вибір кроку (РК-4)\n"
                 r"$\varepsilon>$EPS$\Rightarrow h/2$;"
                 r"$\quad\varepsilon\leq$EPS$/16\Rightarrow 2h$", fontsize=12)

    # --- Ліворуч: розв'язок ---
    ax = axes[0]
    ax.plot(xs_ex, y_exact(xs_ex), color=C_EXACT, lw=2.8, label='Точний y(x)')
    ax.plot(xs_fix, ys_fix, color=C_RK4, lw=1.6, ls='--', marker='o',
            ms=5, label=f'РК-4 фікс. h={H}')
    ax.plot(xs_ad, ys_ad, color=C_ADAPT, lw=1.8, ls='-.', marker='D',
            ms=6, label='РК-4 адаптивний')
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Розв'язок")
    ax.legend(fontsize=8)

    # --- По центру: крок h(x) ---
    ax = axes[1]
    if len(hs_ad) > 0:
        ax.step(xs_ad[1:], hs_ad, color=C_STEP, lw=2.5, where='post',
                label='Адаптивний h(x)')
        ax.axhline(H,     color='orange',  lw=1.5, ls='--', label=f'Фікс. h={H}')
        ax.axhline(h_opt, color=C_HOPT,   lw=1.5, ls=':',
                   label=f'h_opt≈{h_opt:.3f}')
        # Позначаємо унікальні значення кроків
        unique_h = sorted(set(np.round(hs_ad, 6)))
        for uh in unique_h:
            ax.axhline(uh, color='gray', lw=0.6, ls='-', alpha=0.4)
        ax.set_ylim(0, max(hs_ad) * 1.5)
    ax.set_xlabel("x"); ax.set_ylabel("h")
    ax.set_title("Залежність кроку h(x)")
    ax.legend(fontsize=8)
    _add_formula_box(ax,
        "ε > EPS   →  h = h/2\n"
        "ε ≤ EPS/16 → h = h·2\n"
        "інакше    →  h = h")

    # --- Праворуч: похибка фікс. vs адаптивний ---
    ax = axes[2]
    err_fix = np.abs(y_exact(xs_fix) - ys_fix)
    err_ad  = np.abs(y_exact(xs_ad)  - ys_ad)
    ax.semilogy(xs_fix, err_fix + 1e-16, color=C_RK4, lw=2, marker='o',
                ms=5, label=f'Фікс. h={H}')
    ax.semilogy(xs_ad, err_ad + 1e-16, color=C_ADAPT, lw=2, marker='D',
                ms=5, label='Адаптивний')
    ax.axhline(EPS, color=C_EPS, lw=1.5, ls='--', label=f'EPS={EPS}')
    ax.set_xlabel("x"); ax.set_ylabel("Похибка (лог.)")
    ax.set_title("Похибка: фікс. vs адаптивний")
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "plot_step9_rk4_adaptive.png")
    if show: plt.show()
    plt.close(fig)


# ============================================================
# ЗВЕДЕНИЙ ОГЛЯД — всі методи на одній фігурі
# ============================================================
def plot_summary(show=True):
    """
    Підсумковий графік: точний розв'язок, Адамс 2-го порядку,
    РК-4 та адаптивний РК-4 — на одному полі.
    Порівняння похибок всіх методів.
    """
    xs_ex = np.linspace(X0, X_END, 600)
    ys_ex = y_exact(xs_ex)

    xs_a,  ys_a,  _  = solve_adams2(X0, Y0, X_END, H)
    xs_r,  ys_r       = solve_rk4(X0, Y0, X_END, H)
    xs_ad, ys_ad, hs_ad = solve_rk4_adaptive(X0, Y0, X_END, H, EPS)

    err_a  = np.abs(y_exact(xs_a)  - ys_a)
    err_r  = np.abs(y_exact(xs_r)  - ys_r)
    err_ad = np.abs(y_exact(xs_ad) - ys_ad)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Підсумок — Порівняння методів (Адамс 2-го пор. vs РК-4)\n"
                 r"$y'=-2y+x,\ y(0)=1,\ h=" + str(H) + r",\ \varepsilon=" + str(EPS) + "$",
                 fontsize=12)

    # --- Ліворуч: криві ---
    ax = axes[0]
    ax.plot(xs_ex, ys_ex, color=C_EXACT, lw=2.8, zorder=5,
            label='Точний y(x)')
    ax.plot(xs_a, ys_a, color=C_ADAMS, lw=1.8, ls='--', marker='o',
            ms=5, label=f'Адамс 2-го пор. (h={H})')
    ax.plot(xs_r, ys_r, color=C_RK4, lw=1.8, ls='-.', marker='s',
            ms=5, label=f'РК-4 (h={H})')
    ax.plot(xs_ad, ys_ad, color=C_ADAPT, lw=1.5, ls=':', marker='D',
            ms=5, label='РК-4 адаптивний')
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Розв'язки всіх методів")
    ax.legend()

    # --- Праворуч: похибки ---
    ax = axes[1]
    ax.semilogy(xs_a,  err_a  + 1e-16, color=C_ADAMS, lw=2, marker='o',
                ms=5, label='Адамс 2-го пор.')
    ax.semilogy(xs_r,  err_r  + 1e-16, color=C_RK4,   lw=2, marker='s',
                ms=5, label='РК-4')
    ax.semilogy(xs_ad, err_ad + 1e-16, color=C_ADAPT,  lw=2, marker='D',
                ms=5, label='РК-4 адаптивний')
    ax.axhline(EPS, color=C_EPS, lw=1.8, ls='--', label=f'EPS={EPS}')
    ax.set_xlabel("x"); ax.set_ylabel("Похибка (лог.)")
    ax.set_title("Локальні похибки всіх методів")
    ax.legend()
    _add_formula_box(ax,
        f"Адамс: O(h²), h={H}\n"
        f"РК-4:  O(h⁴), h={H}\n"
        "РК-4 адапт.: ε≤EPS")

    plt.tight_layout()
    _save(fig, "plot_summary.png")
    if show: plt.show()
    plt.close(fig)


# ============================================================
# ГОЛОВНА ФУНКЦІЯ
# ============================================================
def build_all_plots(show=False):
    print("\n" + "="*62)
    print("  Побудова ВСІХ графіків лабораторної роботи №10")
    print("="*62)

    plots = [
        (plot_step1_exact,           "Крок 1 — Точний розв'язок"),
        (plot_step2_adams_solution,  "Крок 2 — Розв'язок Адамса 2-го пор."),
        (plot_step3_adams_local_error,"Крок 3 — Локальна похибка Адамса"),
        (plot_step4_adams_runge,     "Крок 4 — Оцінка похибки Адамса (Рунге)"),
        (plot_step5_adams_adaptive,  "Крок 5 — Адаптивний крок Адамса"),
        (plot_step6_rk4_solution,    "Крок 6 — Розв'язок РК-4"),
        (plot_step7_rk4_error,       "Крок 7 — Локальна похибка РК-4"),
        (plot_step8_rk4_runge,       "Крок 8 — Оцінка похибки РК-4 (Рунге)"),
        (plot_step9_rk4_adaptive,    "Крок 9 — Адаптивний крок РК-4"),
        (plot_summary,               "Підсумок — Порівняння методів"),
    ]

    for func, label in plots:
        print(f"\n▶ {label}")
        func(show=show)

    print("\n" + "="*62)
    print("  Всі графіки побудовано та збережено!")
    print("="*62)
    files = [
        "plot_step1_exact.png",
        "plot_step2_adams_solution.png",
        "plot_step3_adams_error.png",
        "plot_step4_adams_runge.png",
        "plot_step5_adams_adaptive.png",
        "plot_step6_rk4_solution.png",
        "plot_step7_rk4_error.png",
        "plot_step8_rk4_runge.png",
        "plot_step9_rk4_adaptive.png",
        "plot_summary.png",
    ]
    for f_name in files:
        print(f"   • {f_name}")


if __name__ == "__main__":
    build_all_plots(show=False)