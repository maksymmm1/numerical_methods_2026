import urllib.request
import ssl
import json
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Завантаження GPS-вузлів та висоти
# -------------------------------
ssl_context = ssl._create_unverified_context()  # Ігноруємо перевірку сертифікатів (для Mac/Windows)

url = ("https://api.open-elevation.com/api/v1/lookup?"
       "locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
       "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|"
       "48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|"
       "48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
       "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|"
       "48.160580,24.500537|48.160250,24.500106")

with urllib.request.urlopen(url, context=ssl_context) as response:
    data = json.loads(response.read().decode())

results = data["results"]

# Табуляція вузлів
print("№ | Latitude | Longitude | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}")


# -------------------------------
# 2. Кумулятивна відстань
# -------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # радіус Землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances = [0]

for i in range(1, len(results)):
    distances.append(distances[-1] + haversine(*coords[i - 1], *coords[i]))

print("\n№ | Distance (m) | Elevation (m)")
for i in range(len(results)):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")


# -------------------------------
# 3. Кубічний сплайн (метод прогонки)
# -------------------------------
def cubic_spline_coefficients(x, y):
    n = len(x) - 1
    h = np.diff(x)
    alpha = [0] + [3 * (y[i + 1] - y[i]) / h[i] - 3 * (y[i] - y[i - 1]) / h[i - 1] for i in range(1, n)]

    l = np.ones(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)

    # Пряма прогонка
    for i in range(1, n):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    # Зворотна прогонка
    b = np.zeros(n)
    c = np.zeros(n + 1)
    d = np.zeros(n)
    a = np.array(y[:n])

    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    return a, b, c[:n], d


a, b, c, d = cubic_spline_coefficients(distances, elevations)

# -------------------------------
# 4. Побудова графіка сплайна
# -------------------------------
xx = np.linspace(distances[0], distances[-1], 500)
yy = np.zeros_like(xx)

for i in range(len(a)):
    mask = (xx >= distances[i]) & (xx <= distances[i + 1])
    dx = xx[mask] - distances[i]
    yy[mask] = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3

plt.plot(distances, elevations, 'o', label='Вузли')
plt.plot(xx, yy, '-', label='Кубічний сплайн')
plt.xlabel("Distance (m)")
plt.ylabel("Elevation (m)")
plt.title("Профіль висоти маршруту (Заросляк → Говерла)")
plt.legend()
plt.show()

# -------------------------------
# 5. Перевірка правильності сплайна
# -------------------------------
spline_at_nodes = []
for i in range(len(distances)):
    for j in range(len(a)):
        if distances[i] >= distances[j] and distances[i] <= distances[j + 1]:
            dx = distances[i] - distances[j]
            val = a[j] + b[j] * dx + c[j] * dx ** 2 + d[j] * dx ** 3
            spline_at_nodes.append(val)
            break

print("\nПеревірка вузлів (сплайн vs реальні дані):")
for i in range(len(distances)):
    print(
        f"{i:2d} | Сплайн: {spline_at_nodes[i]:8.2f} | Реально: {elevations[i]:8.2f} | Похибка: {spline_at_nodes[i] - elevations[i]:.2e}")

# -------------------------------
# 6. Додатково: характеристики маршруту
# -------------------------------
total_length = distances[-1]
total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, len(elevations)))
total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, len(elevations)))
mass = 80
g = 9.81
energy_j = mass * g * total_ascent

print("\nХарактеристики маршруту:")
print(f"Загальна довжина (м): {total_length:.2f}")
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
print(f"Сумарний спуск (м): {total_descent:.2f}")
print(f"Механічна робота (Дж): {energy_j:.2f}")
print(f"Механічна робота (кДж): {energy_j / 1000:.2f}")
print(f"Енергія (ккал): {energy_j / 4184:.2f}")

# -------------------------------
# 7. Градієнт маршруту (%)
# -------------------------------
grad_full = np.gradient(yy, xx) * 100
print("\nГрадієнт маршруту:")
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")
print(f"Ділянки з крутизною > 15%: {np.sum(np.abs(grad_full) > 15)} відрізків")