"""
horner.py
Схема Горнера для обчислення значення многочлена і його похідної.
Метод Ньютона для знаходження дійсних коренів алгебраїчного рівняння.

Формат коефіцієнтів: coeffs = [a0, a1, ..., am]
  a0 — вільний член, am — коефіцієнт при x^m

Схема Горнера для F(x) = a0 + a1*x + ... + am*x^m:
  Записуємо у вигляді: F(x) = a0 + x*(a1 + x*(a2 + ... + x*am))
  Від старшого до молодшого:
    b_m = a_m
    b_i = a_i + x * b_{i+1}   (i = m-1, ..., 0)
    F(x) = b_0

  Похідна (другий прохід):
    c_{m-1} = b_m
    c_i     = b_{i+1} + x * c_{i+1}   (i = m-2, ..., 0)
    F'(x)   = c_0
"""


def read_coefficients(filename):
    """
    Зчитує коефіцієнти многочлена з текстового файлу.
    Формат рядка: a0 a1 a2 ... am  (від вільного члена до старшого)
    Рядки, що починаються з '#', ігноруються.
    Повертає список float.
    """
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return list(map(float, line.split()))
    raise ValueError(f"Коефіцієнти не знайдено у файлі '{filename}'")


def horner_eval(coeffs, x):
    """
    Обчислює F(x) та F'(x) за схемою Горнера.
    coeffs = [a0, a1, ..., am]  (a0 — вільний член)

    Алгоритм (обробляємо від старшого коефіцієнта):
      b[m] = am
      b[i] = a[i] + x * b[i+1]   для i = m-1..0
      F(x) = b[0]

      c[m-1] = b[m]
      c[i]   = b[i+1] + x * c[i+1]   для i = m-2..0
      F'(x)  = c[0]
    """
    m = len(coeffs) - 1   # степінь многочлена

    # Перший прохід — F(x)
    # Обробляємо від am до a0 (зберігаємо b в прямому порядку індексів)
    b = [0.0] * (m + 1)
    b[m] = coeffs[m]
    for i in range(m - 1, -1, -1):
        b[i] = coeffs[i] + x * b[i + 1]
    Fx = b[0]

    # Другий прохід — F'(x)
    if m == 0:
        return Fx, 0.0

    c = [0.0] * m
    c[m - 1] = b[m]
    for i in range(m - 2, -1, -1):
        c[i] = b[i + 1] + x * c[i + 1]
    dFx = c[0]

    return Fx, dFx


def newton_horner(coeffs, x0=1.0, eps=1e-10, max_iter=1000):
    """
    Знаходить дійсний корінь многочлена методом Ньютона зі схемою Горнера.
    Повертає (корінь, кількість_ітерацій).
    """
    x = x0
    for i in range(1, max_iter + 1):
        Fx, dFx = horner_eval(coeffs, x)
        if abs(dFx) < 1e-14:
            raise ZeroDivisionError(f"F'({x}) ≈ 0, спробуйте інший x0")
        xn = x - Fx / dFx
        Fxn, _ = horner_eval(coeffs, xn)
        if abs(Fxn) < eps and abs(xn - x) < eps:
            return xn, i
        x = xn
    return x, max_iter


if __name__ == "__main__":
    coeffs = read_coefficients("coefficients.txt")
    print(f"Коефіцієнти: {coeffs}")

    # Перевірка значень
    print(f"F(-2)  = {horner_eval(coeffs, -2.0)[0]:.6f}  (очікується 0)")
    print(f"F(0.5) = {horner_eval(coeffs,  0.5)[0]:.6f}")

    root, iters = newton_horner(coeffs, x0=-2.5)
    print(f"Дійсний корінь: {root:.10f}  (ітерацій: {iters})")
    val, _ = horner_eval(coeffs, root)
    print(f"F(root) = {val:.2e}")