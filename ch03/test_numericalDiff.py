import numpy as np
import matplotlib.pylab as plt


def function_1(x):
    return 0.01 * x**2 + 0.1 * x


def function_2(x):
    return np.sum(x**2)


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# print(numerical_diff(function_1, 5))
# print(numerical_diff(function_1, 10))

# x = np.arange(0.0, 20.0, 0.1)  # 0에서 20까지 0.1 간격의 배열 x를 만든다.
# y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x, y)
# plt.show()

x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x0, x1)

X = X.flatten()
Y = Y.flatten()

plt.figure()
plt.quiver(X, Y, -1, -1, angles="xy", color="#666666")
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel("x0")
plt.ylabel("x1")
plt.grid()
plt.legend()
plt.draw()
plt.show()
