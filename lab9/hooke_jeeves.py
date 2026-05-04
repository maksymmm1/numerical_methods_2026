"""
Метод Хука-Дживса (Hooke-Jeeves) багатовимірної оптимізації.
Лабораторна робота №9
"""

import numpy as np
from typing import Callable, Tuple, List


def hooke_jeeves(
    f: Callable,
    x0: np.ndarray,
    h0: float = 0.5,
    alpha: float = 2.0,
    beta: float = 0.5,
    eps1: float = 1e-6,
    eps2: float = 1e-6,
    max_iter: int = 10000
) -> Tuple[np.ndarray, float, List[np.ndarray], int]:
    """
    Метод Хука-Дживса для знаходження мінімуму функції f(x).

    Параметри:
        f      - цільова функція f: R^n -> R
        x0     - початкове наближення (базисна точка)
        h0     - початкова величина кроку
        alpha  - коефіцієнт збільшення кроку (не використовується в класичному варіанті)
        beta   - коефіцієнт зменшення кроку (звичайно 0.5)
        eps1   - критерій зупинки по значенню функції
        eps2   - критерій зупинки по кроку
        max_iter - максимальна кількість ітерацій

    Повертає:
        x_min     - знайдена точка мінімуму
        f_min     - значення функції в мінімумі
        trajectory - траєкторія спуску (список базисних точок)
        steps     - кількість кроків
    """
    n = len(x0)
    x_base = np.array(x0, dtype=float)   # Базисна точка
    h = np.full(n, h0, dtype=float)       # Кроки по кожній змінній
    trajectory = [x_base.copy()]
    steps = 0

    def exploratory_search(x_start, step):
        """Досліджуючий пошук навколо точки x_start."""
        x = x_start.copy()
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += step[i]
            if f(x_plus) < f(x):
                x = x_plus
            else:
                x_minus = x.copy()
                x_minus[i] -= step[i]
                if f(x_minus) < f(x):
                    x = x_minus
        return x

    for iteration in range(max_iter):
        steps += 1

        # Досліджуючий пошук з базисної точки
        x_current = exploratory_search(x_base, h)

        if f(x_current) < f(x_base):
            # Пошук по зразку
            x_pattern = x_current + (x_current - x_base)
            x_new = exploratory_search(x_pattern, h)

            prev_base = x_base.copy()
            if f(x_new) < f(x_current):
                x_base = x_new.copy()
            else:
                x_base = x_current.copy()

            trajectory.append(x_base.copy())

            # Критерії зупинки
            delta_f = abs(f(x_base) - f(prev_base))
            delta_x = np.linalg.norm(x_base - prev_base)
            if delta_f < eps1 and delta_x < eps2:
                break
        else:
            # Зменшуємо крок
            h = h * beta
            if np.all(h < eps2):
                break

    return x_base, f(x_base), trajectory, steps