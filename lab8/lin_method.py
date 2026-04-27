"""
lin_method.py
Метод Ліна для знаходження комплексно-спряжених коренів
алгебраїчного рівняння.

Алгоритм (за методичкою):
  Розкладаємо F(x) = (x^2 + p*x + q) * Q(x) + r1*x + r0
  де p = -2*alpha, q = alpha^2 + beta^2

  Рекурентність ділення (від старшого коефіцієнта):
    b[0] = a[0]
    b[1] = a[1] - p*b[0]
    b[i] = a[i] - p*b[i-1] - q*b[i-2]

  Залишок: r1 = b[m-1], r0 = b[m]

  Уточнення (з умов r0=0, r1=0):
    q_new = a[m] / b[m-2]
    p_new = (a[m-1] - q_new*b[m-3]) / b[m-2]
"""

import math


def _divide_by_quadratic(a_rev, p, q):
    """
    Ділить многочлен (від старшого a_rev[0]=am до молодшого)
    на (x^2 + p*x + q).
    Повертає масив b: b[m-1], b[m] — залишок (коефіцієнти при x та вільний).
    """
    m = len(a_rev) - 1
    b = [0.0] * (m + 1)
    b[0] = a_rev[0]
    if m >= 1:
        b[1] = a_rev[1] - p * b[0]
    for i in range(2, m + 1):
        b[i] = a_rev[i] - p * b[i - 1] - q * b[i - 2]
    return b


def lin_method(coeffs, alpha0=0.5, beta0=0.5, eps=1e-10, max_iter=1000):
    """
    Метод Ліна для знаходження комплексно-спряжених коренів.

    Вхід:
      coeffs  — [a0, a1, ..., am]  (a0 — вільний член, am — старший)
      alpha0, beta0 — початкові наближення
      eps     — задана точність

    Вихід:
      (alpha, beta, ітерацій)  — корені: alpha ± beta*i
    """
    a_rev = list(reversed(coeffs))   # a_rev[0]=am, a_rev[m]=a0
    m = len(a_rev) - 1

    if m < 2:
        raise ValueError("Степінь многочлена має бути >= 2")

    alpha = alpha0
    beta  = beta0

    for iteration in range(1, max_iter + 1):
        p = -2.0 * alpha
        q = alpha ** 2 + beta ** 2

        b = _divide_by_quadratic(a_rev, p, q)

        r0 = b[m]        # вільний член залишку
        r1 = b[m - 1]    # коефіцієнт при x у залишку

        if abs(r0) < eps and abs(r1) < eps:
            return alpha, beta, iteration

        b2 = b[m - 2]
        b3 = b[m - 3] if m >= 3 else 0.0

        if abs(b2) < 1e-14:
            alpha += 0.1
            beta   = max(abs(beta) + 0.1, 0.1)
            continue

        a0 = a_rev[m]        # вільний член вихідного рівняння
        a1 = a_rev[m - 1]

        q_new = a0 / b2
        p_new = (a1 - q_new * b3) / b2

        alpha_new = -p_new / 2.0
        disc      = q_new - alpha_new ** 2

        if disc < -1e-8:
            alpha = alpha_new
            beta  = max(abs(beta) * 0.9, 1e-6)
            continue

        beta_new = math.sqrt(max(disc, 0.0))

        if abs(alpha_new - alpha) < eps and abs(beta_new - beta) < eps:
            return alpha_new, beta_new, iteration

        alpha = alpha_new
        beta  = beta_new

    return alpha, beta, max_iter


if __name__ == "__main__":
    from horner import read_coefficients
    coeffs = read_coefficients("coefficients.txt")
    print(f"Коефіцієнти: {coeffs}")
    alpha, beta, iters = lin_method(coeffs, alpha0=0.0, beta0=1.0)
    print(f"Комплексні корені: {alpha:.10f} +/- {beta:.10f}i  (ітерацій: {iters})")