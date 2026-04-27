"""
tabulation.py
Табуляція трансцендентної функції F(x) на відрізку [a, b] з кроком h.
Результати записуються у текстовий файл.
Знаходяться наближені значення коренів (абсциси перетину з віссю x).
"""


def tabulate_function(F, a, b, h=0.1, filename="tabulation.txt"):
    """
    Табулює функцію F на [a, b] з кроком h.
    Зберігає таблицю у filename.
    Повертає список пар (x, F(x)).
    """
    nodes = []
    x = a
    while x <= b + 1e-12:
        nodes.append((round(x, 10), F(x)))
        x = round(x + h, 10)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{'x':>12}  {'F(x)':>20}\n")
        f.write("-" * 36 + "\n")
        for xi, yi in nodes:
            f.write(f"{xi:>12.6f}  {yi:>20.10f}\n")

    print(f"[tabulation] Збережено {len(nodes)} вузлів у '{filename}'")
    return nodes


def find_approximate_roots(nodes):
    """
    Знаходить наближені корені як середини відрізків зміни знаку F(x).
    Повертає список наближених значень коренів.
    """
    roots = []
    for i in range(len(nodes) - 1):
        x0, y0 = nodes[i]
        x1, y1 = nodes[i + 1]
        if y0 * y1 < 0:
            x_approx = round((x0 + x1) / 2.0, 6)
            roots.append(x_approx)
    return roots


if __name__ == "__main__":
    import math

    def F(x):
        return math.cos(x) - x

    nodes = tabulate_function(F, -2.0, 2.0, h=0.1)
    roots = find_approximate_roots(nodes)
    print(f"Наближені корені: {roots}")