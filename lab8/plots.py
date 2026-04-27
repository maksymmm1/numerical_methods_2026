import matplotlib.pyplot as plt
import numpy as np
import math

# Імпортуємо методи для порівняння швидкості
from iterative_methods import (simple_iteration, newton_method,
                               chebyshev_method, chord_method,
                               parabola_method, inverse_interpolation)


def build_all_plots(output_dir="plots"):
    # Створюємо одне велике вікно з 3-ма вертикальними підграфіками
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    fig.subplots_adjust(hspace=0.4)  # Відступ між графіками

    # --- 1. Графік трансцендентної функції ---
    def F_np(x):
        return np.cos(x) + 0.5 * x - 1.0

    def F(x):
        return math.cos(x) + 0.5 * x - 1.0

    def dF(x):
        return -math.sin(x) + 0.5

    def d2F(x):
        return -math.cos(x)

    x_vals = np.linspace(-1, 5, 500)
    ax1.plot(x_vals, F_np(x_vals), label=r'$F(x) = \cos(x) + 0.5x - 1$', color='royalblue', linewidth=2)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.axvline(0, color='black', linewidth=1)

    roots = [1.15, 3.65]  # Приблизні точки
    ax1.scatter(roots, [0, 0], color='red', s=50, zorder=5, label='Корені')
    ax1.set_title('Локалізація коренів трансцендентної функції')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- 2. Порівняння ітерацій ---
    x0 = 1.15
    eps = 1e-10
    methods = [
        ("Проста ітерація", simple_iteration),
        ("Ньютона", newton_method),
        ("Чебишева", chebyshev_method),
        ("Хорд", chord_method),
        ("Парабол", parabola_method),
        ("Звор. інтерп.", inverse_interpolation)
    ]

    names, iters = [], []
    for name, method in methods:
        try:
            _, count = method(F, dF, d2F, x0, eps)
            names.append(name)
            iters.append(count)
        except:
            continue

    bars = ax2.bar(names, iters, color='teal', alpha=0.8)
    ax2.set_ylabel('Кількість ітерацій')
    ax2.set_title(f'Швидкість збіжності методів (для кореня x≈1.15, eps={eps})')
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), int(bar.get_height()), ha='center', va='bottom')

    # --- 3. Графік алгебраїчного рівняння ---
    def P_np(x):
        return x ** 3 + x ** 2 + x + 6

    x_p = np.linspace(-4, 2, 500)
    ax3.plot(x_p, P_np(x_p), label=r'$x^3 + x^2 + x + 6 = 0$', color='crimson')
    ax3.axhline(0, color='black', linewidth=1)
    ax3.scatter([-2], [0], color='black', label='Дійсний корінь x=-2')
    ax3.set_title('Алгебраїчне рівняння')
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.legend()

    # Вивід вікна
    print("[plots] Вікно з графіками відкрито.")
    plt.show()