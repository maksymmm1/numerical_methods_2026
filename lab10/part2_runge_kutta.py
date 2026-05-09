# part2_runge_kutta.py
# ============================================================
# ЧАСТИНА 2. Метод Рунге-Кутта четвертого порядку
# ============================================================
# Хід роботи (кроки 6–9 з методички):
#
# Крок 6: Скласти функцію чисельного розв'язку рівняння
#          методом Рунге-Кутта 4-го порядку зі заданим кроком h.
#
# Крок 7: Скласти функцію обчислення локальної похибки
#          delta(x) = |y_exact(x) - y_rk4(x)|
#          та дослідити залежність похибки від h.
#
# Крок 8: Скласти функцію обчислення похибки по методу Рунге.
#          Оцінити необхідну величину кроку для заданої точності.
#
# Крок 9: Скласти функцію автоматичного вибору кроку.
#          Побудувати графік h(x).
#
# ------------------------------------------------------------
# ФОРМУЛИ (з методички, с.6 — явний метод РК 4-го порядку):
#
#   k1 = f(x_n,         y_n)
#   k2 = f(x_n + h/2,   y_n + h/2 * k1)
#   k3 = f(x_n + h/2,   y_n + h/2 * k2)
#   k4 = f(x_n + h,     y_n + h * k3)
#
#   y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
#
# Оцінка похибки (метод Рунге, порядок s=4):
#   eps_n = |y_{n+1}(h) - y_{n+1}(h/2)| / (2^4 - 1)
#         = |y_{n+1}(h) - y_{n+1}(h/2)| / 15
#
# Умова вибору кроку (з методички):
#   Константа C = 2^s = 16
#   якщо eps > EPS         → h = h / 2  (зменшити)
#   якщо eps <= EPS / C    → h = h * 2  (збільшити)
#   якщо EPS/C < eps <= EPS → h не змінювати
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from exact_solution import f, y_exact, X0, Y0, X_END, H, EPS


# ----------------------------------------------------------
# КРОК 6: Один крок та повний розв'язок методом РК-4
# ----------------------------------------------------------
def rk4_step(x, y, h):
    """
    Один крок методу Рунге-Кутта 4-го порядку.

    ФОРМУЛИ (з методички):
      k1 = f(x_n,         y_n)
      k2 = f(x_n + h/2,   y_n + h/2 * k1)
      k3 = f(x_n + h/2,   y_n + h/2 * k2)
      k4 = f(x_n + h,     y_n + h * k3)
      y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    k1 = f(x,           y)
    k2 = f(x + h / 2.0, y + h / 2.0 * k1)
    k3 = f(x + h / 2.0, y + h / 2.0 * k2)
    k4 = f(x + h,       y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_rk4(x0=X0, y0=Y0, x_end=X_END, h=H):
    """
    Повний розв'язок задачі Коші методом Рунге-Кутта 4-го порядку.
    Рівномірна сітка з кроком h.

    Повертає: xs, ys — масиви вузлів та значень розв'язку.
    """
    xs = [x0]
    ys = [y0]
    x, y = x0, y0
    while x + h <= x_end + 1e-12:
        y = rk4_step(x, y, h)
        x = x + h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# ----------------------------------------------------------
# КРОК 7: Локальна похибка через точний розв'язок
# ----------------------------------------------------------
def local_error_exact(xs, ys):
    """
    delta_n = |y_exact(x_n) - y_n|

    Дослідження залежності похибки від h:
    викликаємо solve_rk4 для різних значень h
    та порівнюємо максимальну похибку.
    """
    return np.abs(y_exact(xs) - ys)


def study_error_vs_h(h_list=None, x0=X0, y0=Y0, x_end=X_END):
    """
    Дослідження залежності максимальної локальної похибки від кроку h.
    Для методу РК-4 очікуємо: max_error ≈ C * h^4.
    """
    if h_list is None:
        h_list = [0.2, 0.1, 0.05, 0.025, 0.01, 0.005]
    hs, max_errs = [], []
    for h in h_list:
        xs, ys = solve_rk4(x0, y0, x_end, h)
        err = local_error_exact(xs, ys)
        hs.append(h)
        max_errs.append(np.max(err))
    return np.array(hs), np.array(max_errs)


# ----------------------------------------------------------
# КРОК 8: Оцінка похибки методом Рунге (порядок s=4)
# ----------------------------------------------------------
def runge_error_estimate_rk4(x0=X0, y0=Y0, x_end=X_END, h=H, s=4):
    """
    Оцінка локальної похибки методом Рунге для РК-4:

      eps_n ≈ |y_{n+1}(h) - y_{n+1}(h/2)| / (2^s - 1)

    де s=4 — порядок методу, тому дільник = 2^4 - 1 = 15.

    Також оцінюємо необхідний крок h_opt для досягнення
    заданої точності EPS:
      h_opt = h * (EPS / eps_max)^{1/s}
    """
    xs_h,  ys_h  = solve_rk4(x0, y0, x_end, h)
    xs_h2, ys_h2 = solve_rk4(x0, y0, x_end, h / 2.0)

    # Спільні вузли: кожен 2-й з тонкої сітки
    n = len(xs_h)
    ys_h2_at_h = ys_h2[::2][:n]

    divisor = 2.0 ** s - 1.0   # = 15 для s=4
    eps = np.abs(ys_h - ys_h2_at_h) / divisor

    eps_max = np.max(eps[1:]) if len(eps) > 1 else eps[0]
    if eps_max > 0:
        h_opt = h * (EPS / eps_max) ** (1.0 / s)
    else:
        h_opt = h

    print(f"\nКрок 8: Оцінка похибки методом Рунге (s={s})")
    print(f"  Поточний крок h = {h}")
    print(f"  Максимальна оцінка похибки: {eps_max:.2e}")
    print(f"  Заданa точність EPS = {EPS}")
    print(f"  Рекомендований крок h_opt ≈ {h_opt:.4f}")
    if eps_max > EPS:
        print(f"  ❌ Крок слід зменшити до h_opt ≈ {h_opt:.4f}")
    else:
        print(f"  ✅ Крок оптимальний або надто малий (похибка < EPS)")

    return xs_h, eps, h_opt


# ----------------------------------------------------------
# КРОК 9: Автоматичний вибір кроку для РК-4
# ----------------------------------------------------------
def solve_rk4_adaptive(x0=X0, y0=Y0, x_end=X_END,
                        h_init=H, eps_target=EPS, s=4):
    """
    Розв'язок задачі Коші методом Рунге-Кутта 4-го порядку
    з автоматичним вибором кроку.

    Алгоритм вибору кроку (з методички):
      Константа C = 2^s (для s=4: C = 16)
      eps = |y(h) - y(h/2)| / (2^s - 1)   — оцінка похибки (Рунге)

      якщо eps > EPS_target          → h = h / 2   (зменшити)
      якщо eps ≤ EPS_target / C      → h = h * 2   (збільшити)
      якщо EPS_target/C < eps ≤ EPS  → h не міняти

    На кожному кроці обчислюємо розв'язок з h і з h/2,
    порівнюємо, приймаємо рішення.
    """
    C = 2.0 ** s          # C = 16 для s=4
    divisor = C - 1.0     # = 15

    xs_out = [x0]
    ys_out = [y0]
    hs_out = []

    x = x0
    y = y0
    h = h_init

    while x < x_end - 1e-12:
        if x + h > x_end:
            h = x_end - x

        # Крок з h
        y_h = rk4_step(x, y, h)

        # Два кроки з h/2
        h2 = h / 2.0
        y_mid = rk4_step(x, y, h2)
        y_h2  = rk4_step(x + h2, y_mid, h2)

        # Оцінка похибки: eps = |y(h) - y(h/2)| / (2^s - 1)
        eps = abs(y_h - y_h2) / divisor

        if eps > eps_target:
            # Зменшуємо крок вдвічі, не рухаємо x
            h = h / 2.0
            continue
        else:
            # Приймаємо крок
            x = x + h
            y = y_h
            xs_out.append(x)
            ys_out.append(y)
            hs_out.append(h)

            # Перевіряємо чи можна збільшити крок
            if eps <= eps_target / C:
                h = min(h * 2.0, x_end - x + 1e-15)

    return np.array(xs_out), np.array(ys_out), np.array(hs_out)


# ----------------------------------------------------------
# Побудова графіків (Частина 2)
# ----------------------------------------------------------
def plot_part2(show=True):
    print("=" * 60)
    print("ЧАСТИНА 2: Метод Рунге-Кутта 4-го порядку")
    print("=" * 60)

    # --- Крок 6: чисельний розв'язок ---
    xs, ys = solve_rk4(X0, Y0, X_END, H)
    ys_true = y_exact(xs)

    print(f"\nКрок 6: Розв'язок РК-4, h = {H}")
    print(f"{'x':>8}  {'y_exact':>14}  {'y_rk4':>14}  {'|error|':>12}")
    print("-" * 54)
    for x, ye, yr in zip(xs, ys_true, ys):
        print(f"{x:>8.4f}  {ye:>14.8f}  {yr:>14.8f}  {abs(ye-yr):>12.2e}")

    # --- Крок 7: локальна похибка та залежність від h ---
    err_exact = local_error_exact(xs, ys)
    h_list = [0.2, 0.1, 0.05, 0.025, 0.01]
    hs_study, max_errs = study_error_vs_h(h_list)

    print(f"\nКрок 7: Залежність max|error| від кроку h:")
    print(f"{'h':>10}  {'max|error|':>14}  {'порядок':>10}")
    print("-" * 38)
    for i, (h_i, e_i) in enumerate(zip(hs_study, max_errs)):
        if i == 0:
            print(f"{h_i:>10.4f}  {e_i:>14.2e}  {'—':>10}")
        else:
            order = np.log2(max_errs[i-1] / e_i) / np.log2(hs_study[i-1] / h_i)
            print(f"{h_i:>10.4f}  {e_i:>14.2e}  {order:>10.2f}")

    # --- Крок 8: оцінка похибки методом Рунге ---
    xs_r, err_runge, h_opt = runge_error_estimate_rk4(X0, Y0, X_END, H, s=4)

    # --- Крок 9: адаптивний крок ---
    xs_ad, ys_ad, hs_ad = solve_rk4_adaptive(X0, Y0, X_END, H, EPS)
    err_ad = local_error_exact(xs_ad, ys_ad)

    print(f"\nКрок 9: Адаптивний метод РК-4")
    print(f"  Кількість вузлів: {len(xs_ad)}")
    print(f"  Кроки використані: h_min={np.min(hs_ad):.4f}, h_max={np.max(hs_ad):.4f}")
    print(f"  Max похибка адаптивного методу: {np.max(err_ad):.2e}")

    # ---- Малювання ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle("Лаб. 10 — Частина 2: Метод Рунге-Кутта 4-го порядку", fontsize=14)

    xs_ex = np.linspace(X0, X_END, 500)

    # Граф. 1: Розв'язок РК-4 vs точний
    ax = axes[0, 0]
    ax.plot(xs_ex, y_exact(xs_ex), 'b-', lw=2, label='Точний y(x)')
    ax.plot(xs, ys, 'ro--', ms=5, label=f'РК-4, h={H}')
    ax.set_title("Крок 6: Розв'язок методом РК-4")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(); ax.grid(True)

    # Граф. 2: Локальна похибка (точна)
    ax = axes[0, 1]
    ax.semilogy(xs, err_exact + 1e-16, 'r-o', ms=5,
                label=r'$\delta_n = |y_{exact} - y_{RK4}|$')
    ax.set_title("Крок 7: Локальна похибка (точна)")
    ax.set_xlabel("x"); ax.set_ylabel("Похибка (лог.)")
    ax.legend(); ax.grid(True)

    # Граф. 3: Залежність max|error| від h (логлог)
    ax = axes[1, 0]
    ax.loglog(hs_study, max_errs, 'bs-', ms=7, label='max|error|(h)')
    # Довідкова лінія O(h^4)
    h_ref = np.array(hs_study)
    ax.loglog(h_ref, max_errs[0] * (h_ref / hs_study[0])**4,
              'g--', label='O(h⁴) — еталон')
    ax.set_title("Крок 7: max|error| vs h (логлог)")
    ax.set_xlabel("h (лог.)"); ax.set_ylabel("max|error| (лог.)")
    ax.legend(); ax.grid(True)

    # Граф. 4: Оцінка похибки методом Рунге
    ax = axes[1, 1]
    ax.semilogy(xs_r, err_runge + 1e-16, 'g-s', ms=5,
                label=r'$\varepsilon_n = |y(h)-y(h/2)|/15$')
    ax.axhline(EPS, color='red', ls='--', label=f'EPS={EPS}')
    ax.set_title("Крок 8: Оцінка похибки (Рунге)")
    ax.set_xlabel("x"); ax.set_ylabel("Похибка (лог.)")
    ax.legend(); ax.grid(True)

    # Граф. 5: Адаптивний розв'язок
    ax = axes[2, 0]
    ax.plot(xs_ex, y_exact(xs_ex), 'b-', lw=2, label='Точний y(x)')
    ax.plot(xs_ad, ys_ad, 'mo--', ms=5, label='РК-4 адаптивний')
    ax.set_title("Крок 9: Адаптивний метод РК-4")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(); ax.grid(True)

    # Граф. 6: Крок h(x) адаптивного методу
    ax = axes[2, 1]
    if len(hs_ad) > 0:
        ax.step(xs_ad[1:], hs_ad, 'b-', where='post', label='h(x) адаптивний')
        ax.axhline(H, color='orange', ls='--', label=f'h початковий={H}')
        ax.axhline(h_opt, color='green', ls=':', label=f'h_opt≈{h_opt:.4f}')
    ax.set_title("Крок 9: Залежність кроку h(x)")
    ax.set_xlabel("x"); ax.set_ylabel("h")
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig("lab10_part2_rk4.png", dpi=150, bbox_inches='tight')
    print("\nГрафік збережено: lab10_part2_rk4.png")
    if show:
        plt.show()


if __name__ == "__main__":
    plot_part2(show=True)