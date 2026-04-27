"""
iterative_methods.py
Ітераційні методи розв'язку нелінійного рівняння F(x) = 0:
  1. Метод простої ітерації
  2. Метод Ньютона
  3. Метод Чебишева
  4. Метод хорд
  5. Метод парабол (Мюллера)
  6. Метод зворотної інтерполяції

Кожна функція повертає (корінь, кількість_ітерацій).
Критерій зупинки: |F(xn)| < eps  ТА  |xn - x_{n-1}| < eps
"""

import math

MAX_ITER = 100_000


def _stop(F, xn, xp, eps):
    """Повертає True якщо обидві умови збіжності виконані."""
    return abs(F(xn)) < eps and abs(xn - xp) < eps


# ─────────────────────────────────────────────
# 1. Метод простої ітерації (релаксація)
# ─────────────────────────────────────────────
def simple_iteration(F, dF, d2F, x0, eps=1e-10):
    """
    x_{n+1} = x_n - tau * F(x_n)
    tau = -1 / F'(x0) — забезпечує збіжність поблизу кореня.
    """
    tau = -1.0 / dF(x0)
    x = x0
    for i in range(1, MAX_ITER + 1):
        xn = x + tau * F(x)
        if _stop(F, xn, x, eps):
            return xn, i
        x = xn
    return x, MAX_ITER


# ─────────────────────────────────────────────
# 2. Метод Ньютона (другий порядок збіжності)
# ─────────────────────────────────────────────
def newton_method(F, dF, d2F, x0, eps=1e-10):
    """
    x_{n+1} = x_n - F(x_n) / F'(x_n)
    """
    x = x0
    for i in range(1, MAX_ITER + 1):
        fx  = F(x)
        dfx = dF(x)
        if abs(dfx) < 1e-14:
            raise ZeroDivisionError("F'(x) ≈ 0")
        xn = x - fx / dfx
        if _stop(F, xn, x, eps):
            return xn, i
        x = xn
    return x, MAX_ITER


# ─────────────────────────────────────────────
# 3. Метод Чебишева (третій порядок збіжності)
# ─────────────────────────────────────────────
def chebyshev_method(F, dF, d2F, x0, eps=1e-10):
    """
    x_{n+1} = x_n - F/F' - (F^2 * F'') / (2*(F')^3)
    """
    x = x0
    for i in range(1, MAX_ITER + 1):
        fx   = F(x)
        dfx  = dF(x)
        d2fx = d2F(x)
        if abs(dfx) < 1e-14:
            raise ZeroDivisionError("F'(x) ≈ 0")
        xn = x - fx / dfx - (fx ** 2 * d2fx) / (2.0 * dfx ** 3)
        if _stop(F, xn, x, eps):
            return xn, i
        x = xn
    return x, MAX_ITER


# ─────────────────────────────────────────────
# 4. Метод хорд (порядок збіжності ≈ 1.618)
# ─────────────────────────────────────────────
def chord_method(F, dF, d2F, x0, eps=1e-10):
    """
    x_{n+1} = x_n - F(x_n) * (x_n - x_{n-1}) / (F(x_n) - F(x_{n-1}))
    """
    xp = x0 - 0.1
    x  = x0
    for i in range(1, MAX_ITER + 1):
        fp = F(xp)
        fc = F(x)
        denom = fc - fp
        if abs(denom) < 1e-14:
            break
        xn = x - fc * (x - xp) / denom
        if _stop(F, xn, x, eps):
            return xn, i
        xp, x = x, xn
    return x, MAX_ITER


# ─────────────────────────────────────────────
# 5. Метод парабол / Мюллера (порядок ≈ 1.84)
# ─────────────────────────────────────────────
def parabola_method(F, dF, d2F, x0, eps=1e-10):
    """
    Метод трьох точок (парабола Мюллера).
    Використовує інтерполяцію параболою через три останні наближення.
    """
    x0_ = x0 - 0.1
    x1  = x0
    x2  = x0 + 0.1

    for i in range(1, MAX_ITER + 1):
        f0, f1, f2 = F(x0_), F(x1), F(x2)

        # Різницеві відношення
        d01  = (f1 - f0) / (x1 - x0_)
        d12  = (f2 - f1) / (x2 - x1)
        d012 = (d12 - d01) / (x2 - x0_)

        w    = d12 + d012 * (x2 - x1)
        disc = w ** 2 - 4.0 * f2 * d012

        if disc < 0:
            disc = 0.0

        sq   = math.sqrt(disc)
        den  = w + sq if abs(w + sq) >= abs(w - sq) else w - sq

        if abs(den) < 1e-14:
            break

        xn = x2 - 2.0 * f2 / den
        if _stop(F, xn, x2, eps):
            return xn, i
        x0_, x1, x2 = x1, x2, xn

    return x2, MAX_ITER


# ─────────────────────────────────────────────
# 6. Метод зворотної інтерполяції (Лагранж)
# ─────────────────────────────────────────────
def inverse_interpolation(F, dF, d2F, x0, eps=1e-10):
    """
    Будуємо інтерполяційний многочлен Лагранжа x = L(y)
    на трьох вузлах і знаходимо x при y = 0.
    """
    x0_ = x0 - 0.1
    x1  = x0
    x2  = x0 + 0.1

    for i in range(1, MAX_ITER + 1):
        y0 = F(x0_)
        y1 = F(x1)
        y2 = F(x2)

        # Формула Лагранжа для y = 0
        d01 = y0 - y1
        d02 = y0 - y2
        d12 = y1 - y2

        if abs(d01) < 1e-14 or abs(d02) < 1e-14 or abs(d12) < 1e-14:
            break

        xn = (x0_ * y1 * y2 / (d01 * d02)
              - x1  * y0 * y2 / (d01 * d12)
              + x2  * y0 * y1 / (d02 * d12))

        if _stop(F, xn, x2, eps):
            return xn, i
        x0_, x1, x2 = x1, x2, xn

    return x2, MAX_ITER