"""
Трансцендентне рівняння : F(x) = cos(x) + 0.5*x - 1 = 0
Відрізок табуляції      : [-1, 5], крок h = 0.1
Два корені:
  x1 ≈ 1.15 — функція СПАДАЄ  (F'(x1) < 0)
  x2 ≈ 3.65 — функція ЗРОСТАЄ (F'(x2) > 0)

Алгебраїчне рівняння   : x^3 + x^2 + x + 6 = 0
  Дійсний корінь       : x = -2
  Комплексні корені     : 0.5 ± 1.658i

Точність               : eps = 1e-10
"""

import math

from tabulation        import tabulate_function, find_approximate_roots
from iterative_methods import (simple_iteration, newton_method,
                                chebyshev_method, chord_method,
                                parabola_method, inverse_interpolation)
from horner            import read_coefficients, newton_horner, horner_eval
from lin_method        import lin_method

# ══════════════════════════════════════════════════════
#  Трансцендентна функція та її похідні
# ══════════════════════════════════════════════════════
def F(x):   return math.cos(x) + 0.5 * x - 1.0
def dF(x):  return -math.sin(x) + 0.5
def d2F(x): return -math.cos(x)

EPS = 1e-10
A, B, H = -1.0, 5.0, 0.1

# ══════════════════════════════════════════════════════
#  Пункт 1. Табуляція функції
# ══════════════════════════════════════════════════════
print("=" * 65)
print("ПУНКТ 1. Табуляція F(x) = cos(x) + 0.5x - 1  на [-1, 5]")
print("=" * 65)

nodes = tabulate_function(F, A, B, H, filename="tabulation.txt")
roots_approx = find_approximate_roots(nodes)

print(f"Наближені корені (зміна знаку): {roots_approx}")

if len(roots_approx) < 2:
    print("ПОМИЛКА: знайдено менше двох коренів. Перевірте відрізок.")
    exit(1)

# ──────────────────────────────────────────────────────
# Визначаємо яке початкове наближення відповідає зростанню,
# яке — спаданню, за знаком F'(x)
# ──────────────────────────────────────────────────────
classified = []
for r in roots_approx:
    behaviour = "зростання" if dF(r) > 0 else "спадання"
    classified.append((r, behaviour))

print()
for r, beh in classified:
    print(f"  x0 ≈ {r:.4f}  —  функція {beh}  (F'(x0) = {dF(r):.4f})")

# Беремо перший корінь зі спаданням і перший зі зростанням
x0_spad  = next(r for r, b in classified if b == "спадання")
x0_zrost = next(r for r, b in classified if b == "зростання")

print(f"\nОбрані початкові наближення:")
print(f"  спадання  : x0 = {x0_spad:.4f}")
print(f"  зростання : x0 = {x0_zrost:.4f}")

# ══════════════════════════════════════════════════════
#  Пункти 2–4. Ітераційні методи уточнення коренів
# ══════════════════════════════════════════════════════
methods = [
    ("Проста ітерація",        simple_iteration),
    ("Ньютона",                newton_method),
    ("Чебишева",               chebyshev_method),
    ("Хорд",                   chord_method),
    ("Парабол",                parabola_method),
    ("Зворотна інтерполяція",  inverse_interpolation),
]

for label, x0 in [("спадання", x0_spad), ("зростання", x0_zrost)]:
    print(f"\n{'='*65}")
    print(f"ПУНКТИ 2–4. Корінь ({label}): x0 = {x0:.4f},  eps = {EPS}")
    print(f"{'='*65}")
    print(f"{'Метод':<26} {'Корінь':>18}  {'|F(x)|':>12}  {'Ітерацій':>9}")
    print("-" * 70)
    for name, method in methods:
        try:
            root, iters = method(F, dF, d2F, x0, EPS)
            print(f"{name:<26} {root:>18.10f}  {abs(F(root)):>12.2e}  {iters:>9}")
        except Exception as e:
            print(f"{name:<26} {'ПОМИЛКА':>18}: {e}")

# ══════════════════════════════════════════════════════
#  Пункти 5–9. Алгебраїчне рівняння
# ══════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("ПУНКТИ 5–9. Алгебраїчне рівняння: x^3 + x^2 + x + 6 = 0")
print(f"  Один дійсний корінь: x = -2")
print(f"  Два комплексні корені: 0.5 ± 1.658i")
print("=" * 65)

# Пункт 6–7 — зчитування коефіцієнтів з файлу
coeffs = read_coefficients("coefficients.txt")
print(f"\nКоефіцієнти з файлу: {coeffs}")

terms = []
for i, c in enumerate(coeffs):
    if   i == 0: terms.append(f"{c}")
    elif i == 1: terms.append(f"{c:+g}·x")
    else:        terms.append(f"{c:+g}·x^{i}")
print("Рівняння: " + " ".join(terms) + " = 0")

# Перевірка значення в кількох точках через horner_eval (пункт 7)
print(f"\nПеревірка horner_eval:")
for xv in [-2.0, 0.0, 1.0]:
    val, deriv = horner_eval(coeffs, xv)
    print(f"  F({xv:5.1f}) = {val:10.4f}   F'({xv:5.1f}) = {deriv:10.4f}")

# Пункт 8 — метод Ньютона зі схемою Горнера
print(f"\n─── Пункт 8: Метод Ньютона зі схемою Горнера ───")
real_root, iters = newton_horner(coeffs, x0=-2.5, eps=EPS)
Fval, _ = horner_eval(coeffs, real_root)
print(f"Дійсний корінь  : {real_root:.10f}")
print(f"|F(root)|       : {abs(Fval):.2e}")
print(f"Ітерацій        : {iters}")

# Пункт 9 — метод Ліна
print(f"\n─── Пункт 9: Метод Ліна (комплексні корені) ───")
alpha, beta, iters_lin = lin_method(coeffs, alpha0=0.5, beta0=1.6, eps=EPS)
print(f"Комплексні корені:")
print(f"  z1 = {alpha:.10f} + {beta:.10f}·i")
print(f"  z2 = {alpha:.10f} - {beta:.10f}·i")
print(f"Ітерацій: {iters_lin}")

# Перевірка через теорему Вієта
# x^3 + x^2 + x + 6: сума коренів = -1, добуток = -6
sum_roots     = real_root + 2 * alpha
product_roots = real_root * (alpha**2 + beta**2)
print(f"\nПеревірка (теорема Вієта):")
print(f"  Сума коренів    = {sum_roots:>10.6f}   (очікується -1.0)")
print(f"  Добуток коренів = {product_roots:>10.6f}   (очікується -6.0)")

print(f"\n{'='*65}")
print("Роботу завершено.")
print("=" * 65)

# ══════════════════════════════════════════════════════
#  Генерація графіків
# ══════════════════════════════════════════════════════
print()
from plots import build_all_plots
build_all_plots(output_dir="plots")