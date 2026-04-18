import numpy as np

# ── параметри ────────────────────────────────────────────────────────────────
N    = 100
X_I  = 2.5      # значення всіх x_i
SEED = 42

# ── генерація матриці A ──────────────────────────────────────────────────────
np.random.seed(SEED)
A = np.random.uniform(-10.0, 10.0, (N, N))

# діагональне домінування → LU-розклад без перестановок завжди існує
for i in range(N):
    A[i, i] = np.sum(np.abs(A[i, :])) + 1.0

# ── обчислення вектора вільних членів: b_i = Σ_{j=1}^{n} a_ij * x_j ────────
x_true = np.full(N, X_I)

B = np.zeros(N)
for i in range(N):
    for j in range(N):
        B[i] += A[i, j] * x_true[j]

# ── запис у файли ────────────────────────────────────────────────────────────
np.savetxt("matrix_A.txt", A, fmt="%.10f")
np.savetxt("vector_B.txt", B, fmt="%.10f")

print(f"Матрицю A ({N}x{N}) збережено у  matrix_A.txt")
print(f"Вектор  B ({N})     збережено у  vector_B.txt")
print(f"Точний розв'язок: x_i = {X_I} для всіх i = 1..{N}")