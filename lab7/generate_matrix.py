"""
Лабораторна робота №8
Генерація матриці A з діагональним переважанням та вектора B.
"""

import numpy as np
import random

def generate_diagonally_dominant_matrix(n: int, seed: int = 42) -> np.ndarray:
    """
    Генерує матрицю розміру n×n з діагональним переважанням.
    Для кожного рядка: |a_ii| > sum_{j≠i} |a_ij|
    """
    random.seed(seed)
    np.random.seed(seed)

    A = np.random.uniform(-10.0, 10.0, size=(n, n))

    # Забезпечуємо діагональне переважання
    for i in range(n):
        row_sum = np.sum(np.abs(A[i])) - abs(A[i][i])
        A[i][i] = row_sum + random.uniform(1.0, 5.0)  # діагональний елемент > sum інших

    return A


def compute_rhs(A: np.ndarray, x_exact: np.ndarray) -> np.ndarray:
    """Обчислює вектор правих частин b = A @ x_exact"""
    return A @ x_exact


def save_matrix(A: np.ndarray, filename: str) -> None:
    """Зберігає матрицю у текстовий файл"""
    n = A.shape[0]
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in A:
            f.write(" ".join(f"{val:.10f}" for val in row) + "\n")
    print(f"Матриця збережена у файл: {filename}")


def save_vector(b: np.ndarray, filename: str) -> None:
    """Зберігає вектор у текстовий файл"""
    with open(filename, 'w') as f:
        f.write(f"{len(b)}\n")
        for val in b:
            f.write(f"{val:.10f}\n")
    print(f"Вектор збережений у файл: {filename}")


def main():
    n = 100  # розмірність системи (n ≤ 100)
    x_exact_value = 2.5

    print(f"=== Генерація системи рівнянь розмірності {n}×{n} ===")
    print(f"Точний розв'язок: x_i = {x_exact_value} для всіх i")

    # Генеруємо матрицю
    A = generate_diagonally_dominant_matrix(n)

    # Точний розв'язок (всі компоненти = 2.5)
    x_exact = np.full(n, x_exact_value)

    # Обчислюємо правий вектор
    b = compute_rhs(A, x_exact)

    # Перевірка діагонального переважання
    dominant = all(
        abs(A[i][i]) > sum(abs(A[i][j]) for j in range(n) if j != i)
        for i in range(n)
    )
    print(f"Діагональне переважання виконується: {dominant}")

    # Зберігаємо у файли
    save_matrix(A, "matrix_A.txt")
    save_vector(b, "vector_b.txt")

    print("\nГенерацію завершено!")


if __name__ == "__main__":
    main()