"""
Утиліти: виведення результатів, запис у файл, форматування таблиць.
Лабораторна робота №9
"""

import numpy as np
import os


def print_header(title: str):
    """Виводить заголовок розділу."""
    line = "=" * 65
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def print_result(name: str, x_min, f_min: float, steps: int, eps: float):
    """Виводить результат оптимізації."""
    print(f"\n  Функція       : {name}")
    print(f"  Знайдений мінімум x* = {np.array2string(x_min, precision=8)}")
    print(f"  f(x*)         = {f_min:.2e}")
    print(f"  Кількість кроків : {steps}")
    print(f"  Точність      : eps = {eps}")


def save_trajectory(filepath: str, trajectory, f_func, label: str = ""):
    """
    Зберігає траєкторію спуску у файл.
    Формат: крок | x1 | x2 | ... | f(x)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fout:
        if label:
            fout.write(f"# {label}\n")
        n = len(trajectory[0])
        header = "Крок\t" + "\t".join([f"x{i+1}" for i in range(n)]) + "\tf(x)\n"
        fout.write(header)
        for step, pt in enumerate(trajectory):
            coords = "\t".join([f"{v:.8f}" for v in pt])
            fout.write(f"{step}\t{coords}\t{f_func(pt):.8e}\n")
    print(f"  Траєкторію збережено: {filepath}")


def compare_with_exact(x_found, exact_solutions, tol=1e-4):
    """Порівнює знайдений розв'язок з точними."""
    for i, x_exact in enumerate(exact_solutions):
        err = np.linalg.norm(x_found - x_exact)
        if err < tol:
            return i, err
    # Знаходимо найближчий
    errs = [np.linalg.norm(x_found - xe) for xe in exact_solutions]
    best = int(np.argmin(errs))
    return best, errs[best]