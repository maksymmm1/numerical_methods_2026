import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

csv_content = """Month,Temp
1,-2
2,0
3,5
4,10
5,15
6,20
7,23
8,22
9,17
10,10
11,5
12,0
13,-10
14,3
15,7
16,13
17,19
18,20
19,22
20,21
21,18
22,15
23,10
24,3"""

with open('data.csv', 'w') as f:
    f.write(csv_content)

def read_data(filename):
    df = pd.read_csv(filename)
    return df['Month'].values, df['Temp'].values


def form_system(x, f, m):
    B = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for l in range(m + 1):
            B[k, l] = np.sum(x ** (k + l))

    C = np.zeros(m + 1)
    for k in range(m + 1):
        C[k] = np.sum(f * (x ** k))
    return B, C


def gauss_solve(A, b):
    n = len(b)
    Ab = np.column_stack((A, b.astype(float)))

    for k in range(n):
        max_row = np.argmax(np.abs(Ab[k:, k])) + k
        Ab[[k, max_row]] = Ab[[max_row, k]]

        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]

    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x_sol[i + 1:n])) / Ab[i, i]
    return x_sol


def get_poly_value(x, coef):
    return sum(c * (x ** i) for i, c in enumerate(coef))


def calculate_variance(f_true, f_approx):
    n = len(f_true)
    return np.sqrt(np.sum((f_approx - f_true) ** 2) / n)


x_nodes, f_nodes = read_data('data.csv')

results = []
for m in range(1, 11):
    B, C = form_system(x_nodes, f_nodes, m)
    coeffs = gauss_solve(B, C)
    f_approx = get_poly_value(x_nodes, coeffs)
    disp = calculate_variance(f_nodes, f_approx)
    results.append({'m': m, 'coeffs': coeffs, 'disp': disp})

optimal_res = min(results, key=lambda x: x['disp'])
m_opt = optimal_res['m']

x_future = np.array([25, 26, 27])
f_future = get_poly_value(x_future, optimal_res['coeffs'])

print("-" * 30)
print(f"{'Ступінь m':<10} | {'Дисперсія':<10}")
print("-" * 30)
for r in results:
    print(f"{r['m']:<10} | {r['disp']:<10.4f}")
print("-" * 30)
print(f"Оптимальний ступінь: m = {m_opt}")
print(f"Прогноз (25, 26, 27 міс.): {np.round(f_future, 2)}")

plt.figure(figsize=(10, 12))

plt.subplot(3, 1, 1)
m_vals = [r['m'] for r in results]
d_vals = [r['disp'] for r in results]
plt.plot(m_vals, d_vals, 'ro-', linewidth=2)
plt.title('Залежність дисперсії від ступеня m')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.scatter(x_nodes, f_nodes, color='black', label='Фактичні дані')
x_fine = np.linspace(1, 27, 200)  # h1 для гладкості [cite: 87]
plt.plot(x_fine, get_poly_value(x_fine, optimal_res['coeffs']), 'b-', label=f'МНК (m={m_opt})')
plt.scatter(x_future, f_future, color='green', label='Прогноз (3 міс.)')
plt.title('Апроксимація та прогноз')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
error = np.abs(f_nodes - get_poly_value(x_nodes, optimal_res['coeffs']))
plt.plot(x_nodes, error, 'm-x', label='|f(x) - phi(x)|')
plt.title('Графік похибки апроксимації')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()