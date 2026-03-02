import csv
import matplotlib.pyplot as plt

# -----------------------------
# Зчитування даних
# -----------------------------
def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        try:
            next(reader)  # пропускаємо заголовок
        except StopIteration:
            raise ValueError(f"Файл '{filename}' порожній або не містить даних.")
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    if not x:
        raise ValueError("Файл містить лише заголовок, даних немає.")
    return x, y


# -----------------------------
# Таблиця розділених різниць
# -----------------------------
def divided_differences(x, y):
    n = len(x)
    table = [y.copy()]

    for j in range(1, n):
        column = []
        for i in range(n - j):
            value = (table[j - 1][i + 1] - table[j - 1][i]) / (x[i + j] - x[i])
            column.append(value)
        table.append(column)

    return table


# -----------------------------
# Поліном Ньютона
# -----------------------------
def newton_predict(x, table, value):
    n = len(x)
    result = table[0][0]

    product = 1
    for i in range(1, n):
        product *= (value - x[i - 1])
        result += table[i][0] * product

    return result


# -----------------------------
# Головна частина
# -----------------------------
x, y = read_data("data.csv")
table = divided_differences(x, y)

prediction = newton_predict(x, table, 6000)

print("Прогноз для n=6000:", prediction)

# Побудова графіка
x_plot = list(range(1000, 17000, 500))
y_plot = [newton_predict(x, table, val) for val in x_plot]

plt.scatter(x, y)
plt.plot(x_plot, y_plot)
plt.xlabel("n")
plt.ylabel("t (ms)")
plt.title("Інтерполяція Ньютона")
plt.show()