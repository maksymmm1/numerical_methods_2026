"""
Лабораторна робота №9
Метод Хука-Дживса багатовимірної оптимізації.
Знаходження мінімуму цільових функцій та розв'язку системи нелінійних рівнянь.
"""

import os
import numpy as np

from hooke_jeeves import hooke_jeeves
from target_functions import TARGET_FUNCTIONS
from nonlinear_system import (
    objective_function, system_equations,
    EXACT_SOLUTIONS, INITIAL_GUESSES
)
from plotter import (
    plot_system_equations,
    plot_objective_and_trajectory,
    plot_convergence,
)
from utils import print_header, print_result, save_trajectory, compare_with_exact

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# КРОК 1: Графіки рівнянь системи
# =========================================================
def step1_plot_system():
    print_header("КРОК 1 — Система нелінійних рівнянь")
    print("""
  Система (m=2):
    f1(x1, x2) = x1^2 + x2^2 - 4 = 0   (коло радіуса 2)
    f2(x1, x2) = x1 * x2  - 1   = 0   (гіпербола)

  Аналітичне розв'язання:
    x1^4 - 4*x1^2 + 1 = 0  =>  x1^2 = 2 ± sqrt(3)
    Маємо 4 дійсних розв'язки:
      x* ≈ ( 1.9319,  0.5176)
      x* ≈ ( 0.5176,  1.9319)
      x* ≈ (-1.9319, -0.5176)
      x* ≈ (-0.5176, -1.9319)
    """)
    plot_system_equations(EXACT_SOLUTIONS,
                          save_path=f"{RESULTS_DIR}/system_plot.png")


# =========================================================
# КРОКИ 2–3: Тестування методу на всіх цільових функціях
# =========================================================
def step3_test_all_functions():
    print_header("КРОКИ 2–3 — Тестування методу Хука-Дживса на цільових функціях")

    EPS1, EPS2 = 1e-8, 1e-8
    H0 = 0.5

    trajectories = {}
    f_funcs      = {}

    for name, meta in TARGET_FUNCTIONS.items():
        f       = meta["func"]
        x0      = np.array(meta["x0"], dtype=float)
        x_exact = np.array(meta["minimum"], dtype=float)

        x_min, f_min, traj, steps = hooke_jeeves(
            f, x0, h0=H0, beta=0.5, eps1=EPS1, eps2=EPS2
        )

        err = np.linalg.norm(x_min - x_exact)
        print_result(name, x_min, f_min, steps, EPS1)
        print(f"  Відхилення від мінімуму: {err:.2e}")

        traj_file = f"{RESULTS_DIR}/trajectory_{name.replace(' ', '_')}.txt"
        save_trajectory(traj_file, traj, f, label=f"Функція: {name}")

        trajectories[name] = traj
        f_funcs[name]      = f

        if len(x0) == 2:
            plot_objective_and_trajectory(
                traj, f,
                title=f"{name} — траєкторія Хука-Дживса",
                save_path=f"{RESULTS_DIR}/traj_{name.replace(' ','_')}.png"
            )

    plot_convergence(trajectories, f_funcs,
                     save_path=f"{RESULTS_DIR}/convergence_all.png")


# =========================================================
# КРОКИ 4–5: Розв'язок системи нелінійних рівнянь
# =========================================================
def step4_solve_system():
    print_header("КРОКИ 4–5 — Розв'язок системи нелінійних рівнянь")

    EPS1 = 1e-12
    EPS2 = 1e-12
    H0   = 0.5

    print("""
  Цільова функція:
    F(x) = f1(x)^2 + f2(x)^2
         = (x1^2 + x2^2 - 4)^2 + (x1*x2 - 1)^2
  Параметри: h0=0.5, beta=0.5, eps1=eps2=1e-12
    """)

    for idx, x0 in enumerate(INITIAL_GUESSES):
        label = f"Розв'язок #{idx+1}"
        print(f"\n  --- {label} ---")
        print(f"  Початкове наближення: x0 = {x0}")

        x_min, f_min, traj, steps = hooke_jeeves(
            objective_function, x0,
            h0=H0, beta=0.5, eps1=EPS1, eps2=EPS2
        )

        residuals = system_equations(x_min)
        sol_idx, err = compare_with_exact(x_min, EXACT_SOLUTIONS, tol=0.5)

        print(f"  Знайдений розв'язок : x* = {np.array2string(x_min, precision=8)}")
        print(f"  F(x*)               = {f_min:.2e}")
        print(f"  Нев'язки            : f1 = {residuals[0]:.2e},  f2 = {residuals[1]:.2e}")
        print(f"  Найближчий точний   : {np.array2string(EXACT_SOLUTIONS[sol_idx], precision=8)}")
        print(f"  Похибка ||x-x*||    : {err:.2e}")
        print(f"  Кроків на траєкторії: {steps}")

        traj_file = f"{RESULTS_DIR}/system_trajectory_{idx+1}.txt"
        save_trajectory(traj_file, traj, objective_function,
                        label=f"Система, {label}, x0={x0}")

        plot_objective_and_trajectory(
            traj, objective_function,
            title=f"F(x) = f1²+f2² — {label} (x0={x0})",
            save_path=f"{RESULTS_DIR}/system_traj_{idx+1}.png"
        )


# =========================================================
# ГОЛОВНА ФУНКЦІЯ
# =========================================================
def main():
    print("\n" + "█" * 65)
    print("  ЛАБОРАТОРНА РОБОТА №9")
    print("  Метод Хука-Дживса багатовимірної оптимізації")
    print("█" * 65)

    step1_plot_system()
    step3_test_all_functions()
    step4_solve_system()

    print_header("ВИКОНАННЯ ЗАВЕРШЕНО")
    print(f"  Усі результати збережено у: {os.path.abspath(RESULTS_DIR)}/\n")


if __name__ == "__main__":
    main()