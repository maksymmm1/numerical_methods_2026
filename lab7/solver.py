"""
Лабораторна робота №8
Ітераційні методи розв'язку СЛАР:
  - Метод простої ітерації
  - Метод Якобі
  - Метод Зейделя (Гауса-Зейделя)
"""

import numpy as np
import time


# ─────────────────────────────────────────────
# Функції зчитування з файлів
# ─────────────────────────────────────────────

def read_matrix(filename: str) -> np.ndarray:
    """Зчитує матрицю з текстового файлу."""
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        A = np.zeros((n, n))
        for i in range(n):
            A[i] = list(map(float, f.readline().split()))
    print(f"Матриця {n}×{n} зчитана з файлу: {filename}")
    return A


def read_vector(filename: str) -> np.ndarray:
    """Зчитує вектор з текстового файлу."""
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        b = np.array([float(f.readline()) for _ in range(n)])
    print(f"Вектор розміру {n} зчитаний з файлу: {filename}")
    return b


# ─────────────────────────────────────────────
# Векторні та матричні операції
# ─────────────────────────────────────────────

def mat_vec_product(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Обчислює добуток матриці на вектор A @ x."""
    n = len(x)
    result = np.zeros(n)
    for i in range(n):
        result[i] = sum(A[i][j] * x[j] for j in range(n))
    return result


def vector_norm(x: np.ndarray) -> float:
    """Обчислює норму вектора (евклідова норма)."""
    return np.sqrt(sum(xi ** 2 for xi in x))


def matrix_norm_1(A: np.ndarray) -> float:
    """Обчислює норму матриці (максимум сум по стовпцях, 1-норма)."""
    n = A.shape[0]
    return max(sum(abs(A[i][j]) for i in range(n)) for j in range(n))


def matrix_norm_inf(A: np.ndarray) -> float:
    """Обчислює норму матриці (максимум сум по рядках, ∞-норма)."""
    return max(sum(abs(A[i][j]) for j in range(A.shape[1])) for i in range(A.shape[0]))


# ─────────────────────────────────────────────
# Ітераційні методи
# ─────────────────────────────────────────────

def simple_iteration(A: np.ndarray, b: np.ndarray,
                     x0: np.ndarray, eps: float = 1e-14,
                     max_iter: int = 10000) -> tuple[np.ndarray, int, float]:
    """
    Метод простої ітерації.
    x^(k+1) = x^(k) - τ * (A @ x^(k) - b)
    де τ = 2 / (λ_max + λ_min) — оптимальний параметр.

    Повертає: (розв'язок, кількість ітерацій, норма нев'язки)
    """
    n = len(b)

    # Оцінюємо спектральні межі через норму для вибору τ
    # Для матриці з діагональним переважанням: τ ~ 1 / ||A||
    norm_A = matrix_norm_inf(A)
    tau = 1.0 / norm_A  # консервативний вибір τ

    x = x0.copy()

    for k in range(1, max_iter + 1):
        Ax = mat_vec_product(A, x)
        residual = Ax - b
        x_new = x - tau * residual

        diff = vector_norm(x_new - x)
        x = x_new

        if diff < eps:
            res_norm = vector_norm(mat_vec_product(A, x) - b)
            return x, k, res_norm

    res_norm = vector_norm(mat_vec_product(A, x) - b)
    print(f"  [Проста ітерація] Не збіглась за {max_iter} ітерацій")
    return x, max_iter, res_norm


def jacobi(A: np.ndarray, b: np.ndarray,
           x0: np.ndarray, eps: float = 1e-14,
           max_iter: int = 10000) -> tuple[np.ndarray, int, float]:
    """
    Метод Якобі.
    x_i^(k+1) = (b_i - Σ_{j≠i} a_ij * x_j^(k)) / a_ii

    Повертає: (розв'язок, кількість ітерацій, норма нев'язки)
    """
    n = len(b)
    x = x0.copy()

    for k in range(1, max_iter + 1):
        x_new = np.zeros(n)

        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        diff = vector_norm(x_new - x)
        x = x_new

        if diff < eps:
            res_norm = vector_norm(mat_vec_product(A, x) - b)
            return x, k, res_norm

    res_norm = vector_norm(mat_vec_product(A, x) - b)
    print(f"  [Якобі] Не збіглась за {max_iter} ітерацій")
    return x, max_iter, res_norm


def seidel(A: np.ndarray, b: np.ndarray,
           x0: np.ndarray, eps: float = 1e-14,
           max_iter: int = 10000) -> tuple[np.ndarray, int, float]:
    """
    Метод Гауса-Зейделя.
    x_i^(k+1) = (b_i - Σ_{j<i} a_ij*x_j^(k+1) - Σ_{j>i} a_ij*x_j^(k)) / a_ii

    Повертає: (розв'язок, кількість ітерацій, норма нев'язки)
    """
    n = len(b)
    x = x0.copy()

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(n):
            # Використовуємо вже оновлені значення x_j^(k+1) для j < i
            s1 = sum(A[i][j] * x[j] for j in range(i))          # оновлені
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))  # старі
            x[i] = (b[i] - s1 - s2) / A[i][i]

        diff = vector_norm(x - x_old)

        if diff < eps:
            res_norm = vector_norm(mat_vec_product(A, x) - b)
            return x, k, res_norm

    res_norm = vector_norm(mat_vec_product(A, x) - b)
    print(f"  [Зейдель] Не збіглась за {max_iter} ітерацій")
    return x, max_iter, res_norm


# ─────────────────────────────────────────────
# Початкове наближення
# ─────────────────────────────────────────────

def initial_approximation(n: int) -> np.ndarray:
    """
    Початкове наближення: x_i^(0) = 1.0 * (1 + i/n)
    відповідно до умови завдання: x(i, n) = 1.0 * (1 + i)
    """
    return np.array([1.0 * (1 + i) for i in range(n)], dtype=float)


# ─────────────────────────────────────────────
# Головна функція
# ─────────────────────────────────────────────

def main():
    eps = 1e-14
    x_exact_value = 2.5

    print("=" * 60)
    print("  Лабораторна робота №8: Ітераційні методи розв'язку СЛАР")
    print("=" * 60)

    # Зчитування даних
    A = read_matrix("matrix_A.txt")
    b = read_vector("vector_b.txt")
    n = len(b)

    # Початкове наближення
    x0 = initial_approximation(n)
    print(f"\nПочаткове наближення: x_i = 1.0*(1+i), i=0..{n-1}")
    print(f"Точне рішення:        x_i = {x_exact_value} для всіх i")
    print(f"Точність eps = {eps}\n")

    x_exact = np.full(n, x_exact_value)

    results = {}

    # ── Метод простої ітерації ──
    print("── Метод простої ітерації ──")
    t0 = time.perf_counter()
    x_si, iters_si, res_si = simple_iteration(A, b, x0, eps)
    t_si = time.perf_counter() - t0
    err_si = vector_norm(x_si - x_exact)
    results["Проста ітерація"] = (iters_si, res_si, err_si, t_si)
    print(f"  Ітерацій:          {iters_si}")
    print(f"  Норма нев'язки:    {res_si:.6e}")
    print(f"  Похибка від точного: {err_si:.6e}")
    print(f"  Час:               {t_si:.3f} с\n")

    # ── Метод Якобі ──
    print("── Метод Якобі ──")
    t0 = time.perf_counter()
    x_jac, iters_jac, res_jac = jacobi(A, b, x0, eps)
    t_jac = time.perf_counter() - t0
    err_jac = vector_norm(x_jac - x_exact)
    results["Якобі"] = (iters_jac, res_jac, err_jac, t_jac)
    print(f"  Ітерацій:          {iters_jac}")
    print(f"  Норма нев'язки:    {res_jac:.6e}")
    print(f"  Похибка від точного: {err_jac:.6e}")
    print(f"  Час:               {t_jac:.3f} с\n")

    # ── Метод Зейделя ──
    print("── Метод Гауса-Зейделя ──")
    t0 = time.perf_counter()
    x_sei, iters_sei, res_sei = seidel(A, b, x0, eps)
    t_sei = time.perf_counter() - t0
    err_sei = vector_norm(x_sei - x_exact)
    results["Зейдель"] = (iters_sei, res_sei, err_sei, t_sei)
    print(f"  Ітерацій:          {iters_sei}")
    print(f"  Норма нев'язки:    {res_sei:.6e}")
    print(f"  Похибка від точного: {err_sei:.6e}")
    print(f"  Час:               {t_sei:.3f} с\n")

    # ── Підсумкова таблиця ──
    print("=" * 60)
    print(f"{'Метод':<22} {'Ітерацій':>10} {'||r||':>14} {'||e||':>14} {'Час(с)':>8}")
    print("-" * 60)
    for name, (iters, res, err, t) in results.items():
        print(f"{name:<22} {iters:>10} {res:>14.4e} {err:>14.4e} {t:>8.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()