import numpy as np

num_x, num_y = map(int, input("Введите количество X и Y через пробел: ").split())

XY = np.zeros((num_x, num_y))

for i in range(num_x):
    for j in range(num_y):
        XY[i, j] = float(input(f"Введите значение для XY[{i + 1}, {j + 1}]: "))

px = np.sum(XY, axis=1)
py = np.sum(XY, axis=0)

px /= np.sum(px)
py /= np.sum(py)

print("Исходные данные:")
for i in range(num_x):
    for j in range(num_y):
        print(XY[i, j], end=" ")
    print()

print("Вероятности:")
for i in range(num_x):
    print(f"p(x{i + 1}) = {px[i]:.3f}")
for j in range(num_y):
    print(f"p(y{j + 1}) = {py[j]:.3f}")

pxy = np.outer(px, py)

print("Перемноженные вероятности:")
print(pxy)

if np.allclose(pxy, XY):
    print("\nНезависимые")
else:
    print("\nЗависимые")

px_given_y = XY / np.sum(XY, axis=0)
py_given_x = XY / np.sum(XY, axis=1)[:, np.newaxis]

print("Условные вероятности:")
for i in range(num_x):
    for j in range(num_y):
        print(f"p(x{i + 1}|y{j + 1}) = {px_given_y[i, j]:.3f}")
print()
for j in range(num_y):
    for i in range(num_x):
        print(f"p(y{j + 1}|x{i + 1}) = {py_given_x[i, j]:.3f}")

Hx = -np.sum(px * np.log2(px))
Hy = -np.sum(py * np.log2(py))
Hxy = -np.sum(XY * np.log2(XY))

print("Энтропия:")
print(f"H(X) = {Hx:.3f}")
print(f"H(Y) = {Hy:.3f}")
print(f"H(XY) = {Hxy:.3f}")

Hx_given_y = -np.sum(px_given_y * np.log2(px_given_y), axis=0)
Hy_given_x = -np.sum(py_given_x * np.log2(py_given_x), axis=1)

print("Частные энтропии по X:")
for j in range(num_y):
    print(f"Hy{j + 1}(X) = {Hx_given_y[j]:.3f}")
print("\nЧастные энтропии по Y:")
for i in range(num_x):
    print(f"Hx{i + 1}(Y) = {Hy_given_x[i]:.3f}")

Hx_y = np.sum([py[j] * Hx_given_y[j] for j in range(num_y)])
Hy_x = np.sum([px[i] * Hy_given_x[i] for i in range(num_x)])

print("Полные условные энтропии:")
print(f"Hy(X) = {Hx_y:.3f}")
print(f"Hx(Y) = {Hy_x:.3f}")