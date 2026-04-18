import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# КРОК 2 — Функції програми
# ═══════════════════════════════════════════════════════════════════════════════

# ── 2а. Читання матриці A з текстового файлу ─────────────────────────────────

def read_matrix(path: str) -> np.ndarray:
    """Зчитує матрицю A з текстового файлу."""
    A = np.loadtxt(path)
    print(f"  Матрицю прочитано з '{path}', розмір {A.shape[0]}x{A.shape[1]}")
    return A


# ── 2б. Читання вектора B з текстового файлу ─────────────────────────────────

def read_vector(path: str) -> np.ndarray:
    """Зчитує вектор B з текстового файлу."""
    B = np.loadtxt(path)
    print(f"  Вектор прочитано з '{path}', довжина {len(B)}")
    return B


# ── 2в. Знаходження LU-розкладу матриці A ────────────────────────────────────

def lu_decompose(A: np.ndarray) -> tuple:
    """
    Знаходить LU-розклад матриці A = L * U, де:
      L — нижня трикутна матриця,
      U — верхня трикутна матриця з одиницями на діагоналі (u_ii = 1).

    Алгоритм (за означенням добутку двох матриць a_ik = Σ_j l_ij * u_jk):

      l_ik = a_ik - Σ_{j=1}^{k-1} l_ij * u_jk,   i >= k
      u_ki = (a_ki - Σ_{j=1}^{k-1} l_kj * u_ji) / l_kk,  i > k
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # u_ii = 1 для всіх i
    for i in range(n):
        U[i, i] = 1.0

    # почергово знаходимо k-й стовпець L та k-й рядок U
    for k in range(n):

        # k-й стовпець матриці L: l_ik, i = k..n
        for i in range(k, n):
            s = 0.0
            for j in range(k):
                s += L[i, j] * U[j, k]
            L[i, k] = A[i, k] - s

        if abs(L[k, k]) < 1e-15:
            raise ValueError(
                f"Нульовий ведучий елемент l_{k}{k} — матриця вироджена!"
            )

        # k-й рядок матриці U: u_ki, i = k+1..n
        for i in range(k + 1, n):
            s = 0.0
            for j in range(k):
                s += L[k, j] * U[j, i]
            U[k, i] = (A[k, i] - s) / L[k, k]

    return L, U


# ── 2г. Запис LU-розкладу у текстові файли ───────────────────────────────────

def save_lu(L: np.ndarray, U: np.ndarray,
            path_L: str = "L.txt", path_U: str = "U.txt") -> None:
    """Записує матриці L та U у текстові файли."""
    np.savetxt(path_L, L, fmt="%.10f")
    np.savetxt(path_U, U, fmt="%.10f")
    print(f"  Матрицю L збережено у '{path_L}'")
    print(f"  Матрицю U збережено у '{path_U}'")


# ── 2д. Розв'язок системи AX = B через LU-розклад ───────────────────────────

def forward_substitution(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Пряма підстановка — розв'язує нижню трикутну систему L * Z = B.

      z_1 = b_1 / l_11
      z_k = (b_k - Σ_{j=1}^{k-1} l_kj * z_j) / l_kk,  k = 2..n
    """
    n = len(B)
    Z = np.zeros(n)
    for k in range(n):
        s = 0.0
        for j in range(k):
            s += L[k, j] * Z[j]
        Z[k] = (B[k] - s) / L[k, k]
    return Z


def back_substitution(U: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Зворотна підстановка — розв'язує верхню трикутну систему U * X = Z.

      x_n = z_n
      x_k = z_k - Σ_{j=k+1}^{n} u_kj * x_j,  k = n-1..1
    """
    n = len(Z)
    X = np.zeros(n)
    X[n - 1] = Z[n - 1]
    for k in range(n - 2, -1, -1):
        s = 0.0
        for j in range(k + 1, n):
            s += U[k, j] * X[j]
        X[k] = Z[k] - s
    return X


def solve_lu(L: np.ndarray, U: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Розв'язує систему A*X = B використовуючи LU-розклад:
      1) L * Z = B  — пряма підстановка
      2) U * X = Z  — зворотна підстановка
    """
    Z = forward_substitution(L, B)
    X = back_substitution(U, Z)
    return X


# ── 2е. Обчислення добутку матриці на вектор ─────────────────────────────────

def mat_vec(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Обчислює добуток матриці A на вектор X: результат[i] = Σ_j a_ij * x_j."""
    n = A.shape[0]
    result = np.zeros(n)
    for i in range(n):
        for j in range(n):
            result[i] += A[i, j] * X[j]
    return result


# ── 2є. Обчислення норми вектора ─────────────────────────────────────────────

def vec_norm(V: np.ndarray) -> float:
    """Обчислює норму вектора: max|v_i| (рівномірна норма / норма Чебишева)."""
    return float(np.max(np.abs(V)))


# ═══════════════════════════════════════════════════════════════════════════════
# КРОК 4 — Оцінка точності розв'язку
# ═══════════════════════════════════════════════════════════════════════════════

def accuracy_estimate(A: np.ndarray, X: np.ndarray, B: np.ndarray) -> float:
    """
    Оцінює точність розв'язку за нормою нев'язки:
      eps = max_{1<=i<=n} | Σ_{j=1}^{n} a_ij * x_j - b_i |
    """
    AX = mat_vec(A, X)
    residual = np.zeros(len(B))
    for i in range(len(B)):
        residual[i] = AX[i] - B[i]
    return vec_norm(residual)


# ═══════════════════════════════════════════════════════════════════════════════
# КРОК 5 — Ітераційне уточнення розв'язку
# ═══════════════════════════════════════════════════════════════════════════════

def iterative_refinement(
    A:        np.ndarray,
    L:        np.ndarray,
    U:        np.ndarray,
    B:        np.ndarray,
    X0:       np.ndarray,
    eps:      float = 1e-14,
    max_iter: int   = 50
) -> tuple:
    """
    Ітераційний метод уточнення розв'язку СЛАР.

    На кожній ітерації:
      1. R  = B - A*X           (вектор нев'язки R = B - B0, де B0 = A*X)
      2. A*ΔX = R  →  ΔX       (через вже готовий LU, без повторного розкладу)
      3. X  = X + ΔX            (уточнений розв'язок)

    Умови завершення:
      ||ΔX||   < eps
      ||A*X-B|| < eps

    Перевага: LU-розклад виконується лише ОДИН раз.
    """
    X = X0.copy()

    print(f"\n  {'Ітер':>5} | {'||ΔX||':>14} | {'||AX-B||':>14}")
    print("  " + "-" * 40)

    for iteration in range(1, max_iter + 1):

        # вектор нев'язки: R = B - A*X
        AX = mat_vec(A, X)
        R  = np.zeros(len(B))
        for i in range(len(B)):
            R[i] = B[i] - AX[i]

        # розв'язуємо A*ΔX = R через LU (без нового розкладу!)
        dX = solve_lu(L, U, R)

        # уточнений розв'язок
        for i in range(len(X)):
            X[i] = X[i] + dX[i]

        # норми для перевірки умов зупинки
        err_dx  = vec_norm(dX)
        err_res = accuracy_estimate(A, X, B)

        print(f"  {iteration:>5} | {err_dx:>14.6e} | {err_res:>14.6e}")

        # перевірка умов закінчення ітераційної процедури
        if err_dx < eps and err_res < eps:
            print(f"\n  Збіжність досягнута за {iteration} ітерацій.")
            return X, iteration

    print(f"\n  Досягнуто максимум ітерацій ({max_iter}).")
    return X, max_iter


# ═══════════════════════════════════════════════════════════════════════════════
# ГОЛОВНА ПРОГРАМА
# ═══════════════════════════════════════════════════════════════════════════════

def line(title: str = "") -> None:
    print("\n" + "=" * 55)
    if title:
        print(f"  {title}")
        print("=" * 55)


if __name__ == "__main__":

    N_TRUE = 100
    X_TRUE = 2.5

    # ── Крок 2: читання даних з файлів ─────────────────────────────────────
    line("КРОК 2 — Читання вхідних даних з файлів")
    A = read_matrix("matrix_A.txt")
    B = read_vector("vector_B.txt")
    n = A.shape[0]

    # ── Крок 2: LU-розклад ─────────────────────────────────────────────────
    line("КРОК 2 — LU-розклад матриці A")
    print("  Виконується LU-розклад...")
    L, U = lu_decompose(A)

    # перевірка коректності: ||L*U - A|| ≈ 0
    LU   = L @ U          # тільки для перевірки, використовуємо numpy
    diff = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff[i, j] = LU[i, j] - A[i, j]
    check_lu = vec_norm(diff.flatten())
    print(f"  Перевірка ||L*U - A|| = {check_lu:.3e}  (має бути ≈ 0)")

    # запис LU у файли
    save_lu(L, U)

    # ── Крок 3: розв'язок СЛАР ─────────────────────────────────────────────
    line("КРОК 3 — Розв'язок системи AX = B через LU-розклад")
    X0 = solve_lu(L, U, B)
    print(f"  Знайдено розв'язок X.")
    print(f"  Перші 5 компонент X: {X0[:5]}")

    # ── Крок 4: оцінка точності ────────────────────────────────────────────
    line("КРОК 4 — Оцінка точності розв'язку")

    eps_res = accuracy_estimate(A, X0, B)
    print(f"  eps = max|Σ a_ij*x_j - b_i| = {eps_res:.6e}")

    x_true  = np.full(n, X_TRUE)
    err_sol = vec_norm(X0 - x_true)
    print(f"  Похибка ||X - x_true||       = {err_sol:.6e}")

    # ── Крок 5: ітераційне уточнення ───────────────────────────────────────
    line("КРОК 5 — Ітераційне уточнення (eps0 = 1e-14)")
    print("  Початкове наближення — розв'язок з кроку 3.")
    X_ref, iters = iterative_refinement(A, L, U, B, X0, eps=1e-14)

    eps_ref = accuracy_estimate(A, X_ref, B)
    err_ref = vec_norm(X_ref - x_true)

    # ── Підсумок ────────────────────────────────────────────────────────────
    line("ПІДСУМОК")
    print(f"  {'Метод':<32} {'||AX-B||':>12}  {'||X-x_true||':>13}")
    print("  " + "-" * 60)
    print(f"  {'LU-розклад (без уточнення)':<32} {eps_res:>12.6e}  {err_sol:>13.6e}")
    print(f"  {'Ітераційне уточнення':<32} {eps_ref:>12.6e}  {err_ref:>13.6e}")
    print(f"\n  Кількість ітерацій уточнення: {iters}")