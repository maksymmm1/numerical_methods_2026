"""
Лабораторна робота №5
Складова квадратурна формула Сімпсона.
Методи підвищення точності. Адаптивний алгоритм.

Функція: f(x) = 50 + 20*sin(pi*x/12) + 5*e^(-0.2*(x-12)^2), x ∈ [0, 24]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# ─────────────────────────────────────────────
# 1. Підінтегральна функція та точний інтеграл
# ─────────────────────────────────────────────

def f(x):
    """Функція навантаження на сервер."""
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)


A, B = 0, 24  # межі інтегрування


def exact_integral(a, b):
    """Точне значення інтегралу через scipy."""
    result, _ = integrate.quad(f, a, b)
    return result


I0 = exact_integral(A, B)
print(f"{'='*55}")
print(f"  Точне значення інтегралу I0 = {I0:.10f}")
print(f"{'='*55}\n")


# ─────────────────────────────────────────────
# 2. Складова квадратурна формула Сімпсона
# ─────────────────────────────────────────────

def simpson(func, a, b, N):
    """
    Обчислює інтеграл func на [a, b] складовою формулою Сімпсона з N вузлами.
    N повинно бути парним.
    """
    if N % 2 != 0:
        N += 1  # N має бути парним
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = func(x)
    # I = h/3 * (f0 + 4*(f1+f3+...) + 2*(f2+f4+...) + fN)
    result = y[0] + y[-1]
    result += 4 * np.sum(y[1:N:2])   # непарні індекси
    result += 2 * np.sum(y[2:N-1:2]) # парні індекси (крім крайніх)
    return result * h / 3


# ─────────────────────────────────────────────
# 3. Дослідження залежності точності від N
# ─────────────────────────────────────────────

print("3. Залежність похибки від числа розбиття N")
print("-" * 45)

N_values = range(10, 1001, 2)
errors = [abs(simpson(f, A, B, N) - I0) for N in N_values]

# Знаходимо N_opt при якому похибка < 1e-12
TARGET_EPS = 1e-12
N_opt = None
eps_opt = None
for N, eps in zip(N_values, errors):
    if eps < TARGET_EPS:
        N_opt = N
        eps_opt = eps
        break

if N_opt is None:
    # якщо не знайшли — беремо мінімальну похибку
    idx = np.argmin(errors)
    N_opt = list(N_values)[idx]
    eps_opt = errors[idx]
    print(f"  Точність 1e-12 не досягнута. Найкраща: N={N_opt}, eps={eps_opt:.2e}")
else:
    print(f"  N_opt = {N_opt}  (перше N, де похибка < 1e-12)")
    print(f"  epsopt = |I(N_opt) - I0| = {eps_opt:.2e}")

print()

# ─────────────────────────────────────────────
# 4. Похибка при N0 ≈ N_opt / 10, кратне 8
# ─────────────────────────────────────────────

print("4. Похибка при N0 ≈ N_opt / 10 (кратне 8)")
print("-" * 45)

N0_approx = max(N_opt // 10, 40)  # не менше 40 для коректної роботи методів
N0 = max(8, (N0_approx // 8) * 8)  # округляємо до кратного 8
I_N0 = simpson(f, A, B, N0)
eps0 = abs(I_N0 - I0)

print(f"  N0     = {N0}")
print(f"  I(N0)  = {I_N0:.10f}")
print(f"  eps0   = |I(N0) - I0| = {eps0:.6e}\n")

# ─────────────────────────────────────────────
# 5. Метод Рунге-Ромберга
# ─────────────────────────────────────────────

print("5. Метод Рунге-Ромберга")
print("-" * 45)

N_half = N0 // 2
I_N0_half = simpson(f, A, B, N_half)

# Уточнення: I_R = I(N0) + (I(N0) - I(N0/2)) / 15
I_R = I_N0 + (I_N0 - I_N0_half) / 15
epsR = abs(I_R - I0)

print(f"  I(N0)    = {I_N0:.10f}")
print(f"  I(N0/2)  = {I_N0_half:.10f}")
print(f"  I_R      = {I_R:.10f}  (уточнене за Рунге-Ромбергом)")
print(f"  epsR     = |I_R - I0| = {epsR:.6e}\n")

# ─────────────────────────────────────────────
# 6. Метод Ейткена
# ─────────────────────────────────────────────

print("6. Метод Ейткена")
print("-" * 45)

# Три різні кроки: h, 2h, 4h → N0, N0/2, N0/4
N1 = N0
N2 = N0 // 2
N3 = N0 // 4

I1 = simpson(f, A, B, N1)
I2 = simpson(f, A, B, N2)
I3 = simpson(f, A, B, N3)

h1, h2, h3 = (B - A) / N1, (B - A) / N2, (B - A) / N3

# Оцінка порядку методу
try:
    ratio = abs((I3 - I2) / (I2 - I1))
    if ratio > 0 and abs(I2 - I1) > 1e-30:
        p = np.log(ratio) / np.log(h3 / h1)
    else:
        p = 4.0  # теоретичний порядок Сімпсона
except (ZeroDivisionError, ValueError):
    p = 4.0

# Уточнене значення (формула Ейткена)
denom = (I3 - I2) - (I2 - I1)
if abs(denom) > 1e-30:
    I_A = I1 - (I2 - I1)**2 / denom
else:
    I_A = I1

epsA = abs(I_A - I0)

print(f"  N1={N1}, N2={N2}, N3={N3}")
print(f"  I(N1) = {I1:.10f}")
print(f"  I(N2) = {I2:.10f}")
print(f"  I(N3) = {I3:.10f}")
print(f"  Оцінка порядку методу p ≈ {p:.2f}  (теор. 4)")
print(f"  I_Aitken = {I_A:.10f}")
print(f"  epsA     = |I_A - I0| = {epsA:.6e}\n")

# ─────────────────────────────────────────────
# 7. Адаптивний алгоритм
# ─────────────────────────────────────────────

print("7. Адаптивний алгоритм (на основі Сімпсона)")
print("-" * 45)

def simpson_interval(func, a, b):
    """Формула Сімпсона на одному відрізку [a, b]."""
    m = (a + b) / 2
    return (b - a) / 6 * (func(a) + 4 * func(m) + func(b))


def adaptive_simpson(func, a, b, tol, depth=0, max_depth=50):
    """
    Рекурсивний адаптивний алгоритм Сімпсона.
    Повертає (значення, кількість обчислень функції).
    """
    m = (a + b) / 2
    I_whole = simpson_interval(func, a, b)
    I_left  = simpson_interval(func, a, m)
    I_right = simpson_interval(func, m, b)
    I_refined = I_left + I_right

    if depth >= max_depth:
        return I_refined, 9

    if abs(I_refined - I_whole) <= 15 * tol:
        return I_refined + (I_refined - I_whole) / 15, 9
    else:
        I_l, c_l = adaptive_simpson(func, a, m, tol / 2, depth + 1, max_depth)
        I_r, c_r = adaptive_simpson(func, m, b, tol / 2, depth + 1, max_depth)
        return I_l + I_r, c_l + c_r + 2


tol_values = [1e-3, 1e-5, 1e-7, 1e-9]
print(f"  {'tol':>10}  {'I_adapt':>18}  {'|err|':>12}  {'#f-calls':>10}")
print(f"  {'-'*55}")
adaptive_results = []
for tol in tol_values:
    I_adapt, n_calls = adaptive_simpson(f, A, B, tol)
    err = abs(I_adapt - I0)
    adaptive_results.append((tol, I_adapt, err, n_calls))
    print(f"  {tol:>10.1e}  {I_adapt:>18.10f}  {err:>12.4e}  {n_calls:>10}")

# ─────────────────────────────────────────────
# 8. Зведена таблиця похибок різних методів
# ─────────────────────────────────────────────

print(f"\n{'='*55}")
print("  Зведення похибок різних методів")
print(f"{'='*55}")
print(f"  {'Метод':<30}  {'|err|':>12}")
print(f"  {'-'*44}")
print(f"  {'Сімпсон (N0=' + str(N0) + ')':<30}  {eps0:>12.4e}")
print(f"  {'Рунге-Ромберг':<30}  {epsR:>12.4e}")
print(f"  {'Ейткен':<30}  {epsA:>12.4e}")
for tol, _, err, _ in adaptive_results:
    print(f"  {'Адаптивний (tol=' + f'{tol:.0e}' + ')':<30}  {err:>12.4e}")
print()

# ─────────────────────────────────────────────
# 9–10. Всі три графіки в одному вікні (3 підграфіки)
# ─────────────────────────────────────────────

x_plot = np.linspace(A, B, 1000)
y_plot = f(x_plot)

methods = [f'Сімпсон\n(N0={N0})', 'Рунге-\nРомберг', 'Ейткен'] + \
          [f'Адапт.\ntol={tol:.0e}' for tol, *_ in adaptive_results]
errs = [eps0, epsR, epsA] + [err for _, _, err, _ in adaptive_results]

fig = plt.figure(figsize=(14, 10))
fig.suptitle('Лабораторна робота №5 — Метод Сімпсона', fontsize=13, fontweight='bold')

# ── Підграфік 1: функція навантаження (верхній лівий) ──
ax1 = fig.add_subplot(2, 2, 1)
ax1.fill_between(x_plot, y_plot, alpha=0.15, color='steelblue')
ax1.plot(x_plot, y_plot, color='steelblue', linewidth=2,
         label=r'$f(x)=50+20\sin\!\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$')
ax1.set_title('Функція навантаження на сервер')
ax1.set_xlabel('Час, x (год)')
ax1.set_ylabel('Навантаження, f(x)')
ax1.grid(True, alpha=0.4)
ax1.legend(fontsize=8)

# ── Підграфік 2: залежність похибки від N (верхній правий) ──
ax2 = fig.add_subplot(2, 2, 2)
ax2.semilogy(list(N_values), errors, color='steelblue', linewidth=1.5)
ax2.axhline(TARGET_EPS, color='red', linestyle='--', label='ε = 1e-12')
if N_opt:
    ax2.axvline(N_opt, color='green', linestyle=':', label=f'N_opt = {N_opt}')
ax2.set_title('Залежність похибки від N (Сімпсон)')
ax2.set_xlabel('N (кількість розбиттів)')
ax2.set_ylabel('ε(N) = |I(N) − I₀|')
ax2.legend(fontsize=9)
ax2.grid(True, which='both', alpha=0.4)

# ── Підграфік 3: порівняння методів (нижній, на всю ширину) ──
ax3 = fig.add_subplot(2, 1, 2)
colors = ['#4e79a7', '#f28e2b', '#e15759'] + ['#76b7b2'] * len(adaptive_results)
bars = ax3.bar(methods, errs, color=colors, edgecolor='white', linewidth=0.8)
ax3.set_yscale('log')
ax3.set_ylabel('Абсолютна похибка |I − I₀|')
ax3.set_title('Порівняння похибок методів')
ax3.grid(True, axis='y', which='both', alpha=0.3)
for bar, err in zip(bars, errs):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
             f'{err:.1e}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('lab5_all_plots.png', dpi=150)
plt.show()

print("Графік збережено: lab5_all_plots.png")