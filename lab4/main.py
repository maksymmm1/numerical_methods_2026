import numpy as np
import matplotlib.pyplot as plt

# 1. Визначення функції та її аналітичної похідної
# -----------------------------------------------

def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)


t0 = 1.0
exact = dM_exact(t0)

print("=" * 55)
print("  Лабораторна робота №4: Чисельне диференціювання")
print("=" * 55)
print(f"\n1. Точне значення похідної M'({t0}) = {exact:.6f}")


# 2. Центральна різниця: дослідження похибки від кроку h
# -----------------------------------------------

def central_diff(f, t, h):
    return (f(t + h) - f(t - h)) / (2 * h)

print("\n2. Залежність похибки від кроку h (центральна різниця):")
print(f"{'h':>12} | {'D(h)':>12} | {'Похибка':>12}")
print("-" * 42)

h_values = [10**(-k) for k in range(1, 8)]
errors = []
deriv_vals = []

for h in h_values:
    d = central_diff(M, t0, h)
    err = abs(d - exact)
    deriv_vals.append(d)
    errors.append(err)
    print(f"{h:>12.6f} | {d:>12.6f} | {err:>12.8f}")

best_idx = int(np.argmin(errors))
h_opt = h_values[best_idx]
d_opt = deriv_vals[best_idx]
err_opt = errors[best_idx]

print(f"\n   Оптимальний крок: h_opt = {h_opt}")
print(f"   D(h_opt) = {d_opt:.6f},  Похибка = {err_opt:.8f}")


# 3. Приймаємо h = 0.01
# -----------------------------------------------

h = 0.01
print(f"\n3. Прийнятий крок: h = {h}")


# 4. Обчислення похідної для двох кроків h і h/2
# -----------------------------------------------

h1 = h
h2 = h / 2

D_h1 = central_diff(M, t0, h1)
D_h2 = central_diff(M, t0, h2)

print(f"\n4. D(h)   = {D_h1:.8f}  (h = {h1})")
print(f"   D(h/2) = {D_h2:.8f}  (h = {h2})")


# 5. Похибка при кроці h
# -----------------------------------------------

err_h1 = abs(D_h1 - exact)
print(f"\n5. Похибка при h = {h1}: E(h) = {err_h1:.8f}")


# 6. Метод Рунге-Ромберга (p=2 для центральної різниці)
# -----------------------------------------------

p = 2
D_RR = D_h2 + (D_h2 - D_h1) / (2**p - 1)
err_RR = abs(D_RR - exact)

print(f"\n6. Метод Рунге-Ромберга:")
print(f"   D_RR = {D_RR:.8f}")
print(f"   Похибка E_RR = {err_RR:.8f}")
print(f"   Зменшення похибки у {err_h1/err_RR:.1f} разів")


# 7. Метод Ейткена (три кроки: h, h/2, h/4)
# -----------------------------------------------

h3 = h / 4
D_h3 = central_diff(M, t0, h3)

err_h3 = abs(D_h3 - exact)

# Оцінка порядку точності за методом Ейткена
numerator_order = np.log(abs(D_h2 - D_h1)) - np.log(abs(D_h3 - D_h2))
denominator_order = np.log(2)
p_aitken = numerator_order / denominator_order

# Уточнене значення похідної за методом Ейткена
denom = D_h3 - 2 * D_h2 + D_h1
if abs(denom) > 1e-15:
    D_aitken = D_h1 - (D_h2 - D_h1)**2 / denom
else:
    D_aitken = D_h3

err_aitken = abs(D_aitken - exact)

print(f"\n7. Метод Ейткена:")
print(f"   D(h)   = {D_h1:.8f}  (h  = {h1})")
print(f"   D(h/2) = {D_h2:.8f}  (h/2 = {h2})")
print(f"   D(h/4) = {D_h3:.8f}  (h/4 = {h3})")
print(f"   Порядок точності p ≈ {p_aitken:.2f}")
print(f"   D_Aitken = {D_aitken:.8f}")
print(f"   Похибка E_Aitken = {err_aitken:.8f}")
print(f"   Зменшення похибки у {err_h1/err_aitken:.1f} разів")

print("\n" + "=" * 55)
print("  Підсумкова таблиця")
print("=" * 55)
print(f"  Точне значення M'({t0}) = {exact:.8f}")
print(f"  {'Метод':<20} | {'Значення':>12} | {'Похибка':>12}")
print("  " + "-" * 50)
print(f"  {'Центральна різниця':<20} | {D_h1:>12.8f} | {err_h1:>12.8f}")
print(f"  {'Рунге-Ромберг':<20} | {D_RR:>12.8f} | {err_RR:>12.8f}")
print(f"  {'Ейткен':<20} | {D_aitken:>12.8f} | {err_aitken:>12.8f}")
print("=" * 55)


# Графіки
# -----------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Чисельне диференціювання M(t) = 50e^{-0.1t} + 5sin(t)", fontsize=14)

# Графік функції та її похідної
t_range = np.linspace(0, 20, 500)
axes[0, 0].plot(t_range, M(t_range), 'b-', label='M(t)')
axes[0, 0].set_title('Функція M(t)')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('M(t)')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(t_range, dM_exact(t_range), 'g-', label="M'(t) аналітично")
axes[0, 1].axvline(x=t0, color='r', linestyle='--', alpha=0.5, label=f't₀={t0}')
axes[0, 1].axhline(y=exact, color='orange', linestyle='--', alpha=0.5, label=f"M'(1)={exact:.3f}")
axes[0, 1].set_title("Похідна M'(t)")
axes[0, 1].set_xlabel('t')
axes[0, 1].set_ylabel("M'(t)")
axes[0, 1].legend()
axes[0, 1].grid(True)

# Залежність похибки від кроку h — тільки "розумний" діапазон (h від 0.1 до 1e-5)
# щоб уникнути похибок округлення при дуже малих h
h_plot = h_values[:6]   # 0.1 ... 1e-6, без 1e-7 де вже росте
e_plot = errors[:6]

axes[1, 0].loglog(h_plot, e_plot, 'ro-', linewidth=2, markersize=7)
axes[1, 0].invert_xaxis()   # великі h зліва → малі h справа (як зменшення кроку)
axes[1, 0].axvline(x=h_opt, color='g', linestyle='--', linewidth=1.5, label=f'h_opt={h_opt}')

# Лінія теоретичного нахилу O(h^2)
h_ref = np.array(h_plot)
ref_line = e_plot[0] * (h_ref / h_ref[0]) ** 2
axes[1, 0].loglog(h_plot, ref_line, 'b--', linewidth=1, alpha=0.6, label='O(h²)')

axes[1, 0].set_title('Похибка від кроку h (log-log)')
axes[1, 0].set_xlabel('h (зменшення →)')
axes[1, 0].set_ylabel('|E(h)|')
axes[1, 0].legend()
axes[1, 0].grid(True, which='both', alpha=0.4)

# Порівняння методів — використовуємо реальні (не нульові) похибки для наочності
# Беремо D(h/4) як базу для Рунге-Ромберга та Ейткена щоб показати прогрес
err_base   = abs(D_h1 - exact)         # центральна різниця h=0.01
err_half   = abs(D_h2 - exact)         # h/2
err_rr_val = abs(D_RR - exact) if abs(D_RR - exact) > 1e-15 else 1e-15
err_ai_val = abs(D_aitken - exact) if abs(D_aitken - exact) > 1e-15 else 1e-15

methods = ['Центр. різниця\n(h=0.01)', 'Центр. різниця\n(h/2)', 'Рунге-Ромберг', 'Ейткен']
method_errors = [err_base, err_half, err_rr_val, err_ai_val]
colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']

bars = axes[1, 1].bar(methods, method_errors, color=colors, edgecolor='black', width=0.5)
axes[1, 1].set_title('Похибки методів')
axes[1, 1].set_ylabel('|E|')
axes[1, 1].set_yscale('log')
axes[1, 1].set_ylim(bottom=1e-16)
for bar, val in zip(bars, method_errors):
    label = f'{val:.2e}' if val > 1e-15 else '< 1e-15'
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                    max(val * 3, 1e-15),
                    label, ha='center', va='bottom', fontsize=8)
axes[1, 1].grid(True, axis='y', which='both', alpha=0.4)

plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nГрафік збережено у results.png")