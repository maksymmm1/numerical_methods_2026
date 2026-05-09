# part1_adams.py
# ============================================================
# ЧАСТИНА 1. Метод прогнозу та корекції Адамса 2-го порядку
# ============================================================
# Хід роботи (кроки 1–5 з методички):
#
# Крок 1: Знайти аналітичний розв'язок на відрізку [x0, x_end]
#          з початковою умовою y(x0) = y0.
#
# Крок 2: Скласти програму чисельного розв'язку методом
#          прогнозу та корекції Адамса 2-го порядку.
#
# Крок 3: Побудувати графік локальної похибки
#          delta(x) = |y_exact(x) - y_numerical(x)|
#
# Крок 4: Побудувати графік похибки за формулою оцінки
#          (метод Рунге), перевірити оптимальність кроку.
#
# Крок 5: Написати програму автоматичного вибору кроку.
#          Побудувати графік h(x).
#
# ------------------------------------------------------------
# ФОРМУЛИ (з методички):
#
# Формула прогнозу 2-го порядку (Адамс-Башфорт):
#   y*_{n+1} = y_n + h/2 * (3*f_n - f_{n-1})
#
# Формула корекції 2-го порядку (Адамс-Молтон):
#   y^{(k+1)}_{n+1} = y_n + h/2 * (f(x_{n+1}, y^{(k)}_{n+1}) + f_n)
#
# Формула модифікації (для зменшення похибки прогнозу):
#   y_mod = y*_{n+1} + 1/2 * (y^{(1)}_{n} - y*_{n})
#
# Кінцеве значення після корекції:
#   y_{n+1} = y^{(iter)}_{n+1}
#
# Оцінка локальної похибки (метод Рунге):
#   eps_n = |y_{n+1}(h) - y_{n+1}(h/2)| / (2^s - 1),  s=2
#
# Умова вибору кроку:
#   якщо eps > EPS         → h = h / 2  (зменшити)
#   якщо eps <= EPS/C      → h = h * 2  (збільшити)
#   інакше                 → h не змінювати
#   де C ≈ 2^s = 4
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from exact_solution import f, y_exact, X0, Y0, X_END, H, EPS


# ----------------------------------------------------------
# Допоміжна функція: 1 крок методу Рунге-Кутта 4-го порядку
# (використовується для «розгону» — знаходження початкових
#  вузлів y0, y1, потрібних для старту методу Адамса)
# ----------------------------------------------------------
def rk4_step(x, y, h):
    """
    Один крок методу Рунге-Кутта 4-го порядку.
    Формули (з методички, для запуску Адамса):
      k1 = f(x,       y)
      k2 = f(x + h/2, y + h/2*k1)
      k3 = f(x + h/2, y + h/2*k2)
      k4 = f(x + h,   y + h*k3)
      y_{n+1} = y_n + h/6*(k1 + 2*k2 + 2*k3 + k4)
    """
    k1 = f(x,           y)
    k2 = f(x + h / 2.0, y + h / 2.0 * k1)
    k3 = f(x + h / 2.0, y + h / 2.0 * k2)
    k4 = f(x + h,       y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ----------------------------------------------------------
# КРОК 2: Метод прогнозу та корекції Адамса 2-го порядку
# ----------------------------------------------------------
def adams2_step(x_prev, y_prev, f_prev,
                x_curr, y_curr, f_curr,
                h, max_iter=10, tol=1e-12):
    """
    Один крок методу прогнозу-корекції Адамса 2-го порядку.

    Вхід:
      x_prev, y_prev, f_prev  — вузол n-1: x_{n-1}, y_{n-1}, f(x_{n-1}, y_{n-1})
      x_curr, y_curr, f_curr  — вузол n:   x_n,     y_n,     f(x_n,     y_n)
      h       — крок сітки
      max_iter— максимальна кількість ітерацій корекції
      tol     — точність ітерацій корекції

    Повертає:
      y_next  — y_{n+1}  (після корекції)
      f_next  — f(x_{n+1}, y_{n+1})

    ФОРМУЛИ:
    -------
    1) Прогноз (Адамс-Башфорт 2-го порядку):
       y*_{n+1} = y_n + h/2 * (3*f_n - f_{n-1})

    2) Корекція (Адамс-Молтон 2-го порядку), ітерації:
       y^{(k+1)}_{n+1} = y_n + h/2 * (f(x_{n+1}, y^{(k)}_{n+1}) + f_n)
       Зупинка: |y^{(k+1)} - y^{(k)}| < tol
    """
    x_next = x_curr + h

    # --- Етап прогнозу ---
    # y*_{n+1} = y_n + h/2 * (3*f_n - f_{n-1})
    y_pred = y_curr + (h / 2.0) * (3.0 * f_curr - f_prev)

    # --- Ітерації корекції ---
    y_corr = y_pred
    for _ in range(max_iter):
        f_corr = f(x_next, y_corr)
        # y^{(k+1)}_{n+1} = y_n + h/2 * (f_{n+1}^{(k)} + f_n)
        y_new = y_curr + (h / 2.0) * (f_corr + f_curr)
        if abs(y_new - y_corr) < tol:
            y_corr = y_new
            break
        y_corr = y_new

    f_next = f(x_next, y_corr)
    return y_corr, f_next, y_pred   # також повертаємо y_pred для оцінки похибки


# ----------------------------------------------------------
# Повний розв'язок методом Адамса 2-го порядку
# ----------------------------------------------------------
def solve_adams2(x0=X0, y0=Y0, x_end=X_END, h=H):
    """
    Розв'язок задачі Коші методом прогнозу-корекції Адамса 2-го порядку
    на рівномірній сітці з кроком h.

    Повертає масиви: xs, ys, y_preds
      xs      — вузли сітки
      ys      — чисельний розв'язок (після корекції)
      y_preds — прогнозовані значення (для оцінки похибки)
    """
    xs = [x0]
    ys = [y0]
    fs = [f(x0, y0)]
    y_preds = [y0]

    # «Розгін»: знаходимо y1 методом Рунге-Кутта 4-го порядку
    x1 = x0 + h
    y1 = rk4_step(x0, y0, h)
    f1 = f(x1, y1)
    xs.append(x1)
    ys.append(y1)
    fs.append(f1)
    y_preds.append(y1)

    # Основні кроки Адамса
    x = x1
    y = y1
    fp = fs[0]   # f_{n-1}
    fc = f1      # f_n

    while x + h <= x_end + 1e-12:
        y_next, f_next, y_pred = adams2_step(
            xs[-2], ys[-2], fs[-2],
            x, y, fc, h
        )
        x = x + h
        y = y_next
        fc = f_next
        xs.append(x)
        ys.append(y)
        fs.append(f_next)
        y_preds.append(y_pred)

    return np.array(xs), np.array(ys), np.array(y_preds)


# ----------------------------------------------------------
# КРОК 3: Локальна похибка через точний розв'язок
# ----------------------------------------------------------
def local_error_exact(xs, ys):
    """
    delta_n = |y_exact(x_n) - y_n|
    """
    return np.abs(y_exact(xs) - ys)


# ----------------------------------------------------------
# КРОК 4: Оцінка похибки методом Рунге
# ----------------------------------------------------------
def runge_error_estimate_adams2(x0=X0, y0=Y0, x_end=X_END, h=H, s=2):
    """
    Оцінка локальної похибки методом Рунге для Адамса 2-го порядку:

      eps_n ≈ |y_{n+1}(h) - y_{n+1}(h/2)| / (2^s - 1)

    де s=2 — порядок методу.
    Для цього розв'язок знаходимо з кроком h та з кроком h/2,
    потім порівнюємо у спільних вузлах.
    """
    xs_h,  ys_h,  _ = solve_adams2(x0, y0, x_end, h)
    xs_h2, ys_h2, _ = solve_adams2(x0, y0, x_end, h / 2.0)

    # Вибираємо спільні вузли (кожен 2-й з сітки h/2)
    # Кількість кроків
    n_common = len(xs_h)
    ys_h2_common = ys_h2[::2][:n_common]

    eps = np.abs(ys_h - ys_h2_common) / (2.0 ** s - 1.0)
    return xs_h, eps


# ----------------------------------------------------------
# КРОК 5: Автоматичний вибір кроку
# ----------------------------------------------------------
def solve_adams2_adaptive(x0=X0, y0=Y0, x_end=X_END,
                           h_init=H, eps_target=EPS, s=2):
    """
    Розв'язок задачі Коші методом Адамса 2-го порядку
    з автоматичним вибором кроку (правило подвоєння/поділу кроку).

    Алгоритм (з методички):
      - Обчислюємо розв'язок з кроком h  → y(h)
      - Обчислюємо розв'язок з кроком h/2 → y(h/2)
      - Локальна похибка: eps = |y(h) - y(h/2)| / (2^s - 1)
      - Якщо eps > EPS           → h = h / 2  (зменшити крок)
      - Якщо eps <= EPS / C      → h = h * 2  (збільшити крок), C = 2^s
      - Інакше                   → крок не змінюємо
      де C ≈ 2^s = 4 для s=2
    """
    C = 2.0 ** s   # C = 4 для s=2

    xs_out = [x0]
    ys_out = [y0]
    hs_out = []    # кроки, що використовувались

    # «Розгін» методом РК4 для отримання y_{n-1} та y_n
    h = h_init
    x_prev = x0
    y_prev = y0
    f_prev = f(x0, y0)

    x_curr = x0 + h
    y_curr = rk4_step(x0, y0, h)
    f_curr = f(x_curr, y_curr)

    xs_out.append(x_curr)
    ys_out.append(y_curr)
    hs_out.append(h)

    while x_curr < x_end - 1e-12:
        # Не виходимо за межу
        if x_curr + h > x_end:
            h = x_end - x_curr

        # Крок з h
        y_h, _, y_pred = adams2_step(
            x_prev, y_prev, f_prev,
            x_curr, y_curr, f_curr, h
        )
        # Два кроки з h/2
        h2 = h / 2.0
        y_h2_a, f_h2_a, _ = adams2_step(
            x_prev, y_prev, f_prev,
            x_curr, y_curr, f_curr, h2
        )
        x_mid = x_curr + h2
        # Для другого кроку потрібні дані з «половинних» вузлів
        # Спрощення: використовуємо РК4 як другий «попередній» вузол
        y_h2_b = rk4_step(x_mid, y_h2_a, h2)

        # Оцінка локальної похибки (метод Рунге):
        # eps = |y(h) - y(h/2)| / (2^s - 1)
        eps = abs(y_h - y_h2_b) / (C - 1.0)

        if eps > eps_target:
            # Зменшуємо крок вдвічі
            h = h / 2.0
            # Повторюємо крок — не рухаємо x вперед
            continue
        else:
            # Приймаємо крок
            x_prev, y_prev, f_prev = x_curr, y_curr, f_curr
            x_curr = x_curr + h
            y_curr = y_h
            f_curr = f(x_curr, y_curr)

            xs_out.append(x_curr)
            ys_out.append(y_curr)
            hs_out.append(h)

            # Перевіряємо чи можна збільшити крок
            if eps < eps_target / C:
                h = min(h * 2.0, x_end - x_curr + 1e-15)

    return np.array(xs_out), np.array(ys_out), np.array(hs_out)


# ----------------------------------------------------------
# Побудова графіків (Частина 1)
# ----------------------------------------------------------
def plot_part1(show=True):
    print("=" * 60)
    print("ЧАСТИНА 1: Метод прогнозу та корекції Адамса 2-го порядку")
    print("=" * 60)

    # --- Крок 1: аналітичний розв'язок ---
    xs_ex = np.linspace(X0, X_END, 500)
    ys_ex = y_exact(xs_ex)

    # --- Крок 2: чисельний розв'язок ---
    xs, ys, y_preds = solve_adams2(X0, Y0, X_END, H)
    ys_true = y_exact(xs)

    print(f"\nКрок h = {H}")
    print(f"{'x':>8}  {'y_exact':>14}  {'y_adams':>14}  {'|error|':>12}")
    print("-" * 54)
    for x, ye, ya in zip(xs, ys_true, ys):
        print(f"{x:>8.4f}  {ye:>14.8f}  {ya:>14.8f}  {abs(ye-ya):>12.2e}")

    # --- Крок 3: локальна похибка (точна) ---
    err_exact = local_error_exact(xs, ys)

    # --- Крок 4: оцінка похибки (Рунге) ---
    xs_r, err_runge = runge_error_estimate_adams2(X0, Y0, X_END, H, s=2)

    # --- Крок 5: адаптивний крок ---
    xs_ad, ys_ad, hs_ad = solve_adams2_adaptive(X0, Y0, X_END, H, EPS)

    # ---- Малювання ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Лаб. 10 — Частина 1: Метод Адамса 2-го порядку", fontsize=14)

    # Графік 1: порівняння точного та чисельного розв'язку
    ax = axes[0, 0]
    ax.plot(xs_ex, ys_ex, 'b-', linewidth=2, label='Точний розв\'язок y(x)')
    ax.plot(xs, ys, 'ro--', markersize=5, label='Адамс 2-го порядку')
    ax.set_title("Розв'язок ДР")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)

    # Графік 2: локальна похибка (через точний розв'язок)
    ax = axes[0, 1]
    ax.semilogy(xs, err_exact + 1e-16, 'r-o', markersize=5,
                label=r'$\delta_n = |y_{exact} - y_{Adams}|$')
    ax.set_title("Крок 3: Локальна похибка (точна)")
    ax.set_xlabel("x")
    ax.set_ylabel("Похибка (лог. масштаб)")
    ax.legend()
    ax.grid(True)

    # Графік 3: оцінка похибки методом Рунге
    ax = axes[1, 0]
    ax.semilogy(xs_r, err_runge + 1e-16, 'g-s', markersize=5,
                label=r'$\varepsilon_n = |y(h) - y(h/2)| / (2^s - 1)$')
    ax.axhline(EPS, color='red', linestyle='--', label=f'Задана точність ε={EPS}')
    ax.set_title("Крок 4: Оцінка похибки (Рунге)")
    ax.set_xlabel("x")
    ax.set_ylabel("Похибка (лог. масштаб)")
    ax.legend()
    ax.grid(True)

    # Графік 4: залежність кроку від x (адаптивний метод)
    ax = axes[1, 1]
    if len(hs_ad) > 0:
        ax.step(xs_ad[1:], hs_ad, 'b-', where='post',
                label='Адаптивний крок h(x)')
        ax.axhline(H, color='orange', linestyle='--', label=f'Початковий h={H}')
    ax.set_title("Крок 5: Адаптивний крок h(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("h")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("lab10_part1_adams.png", dpi=150, bbox_inches='tight')
    print("\nГрафік збережено: lab10_part1_adams.png")
    if show:
        plt.show()


if __name__ == "__main__":
    plot_part1(show=True)