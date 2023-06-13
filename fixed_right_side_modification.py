import numpy as np


def power_method():
    ctr = 0
    x = x_0
    lam_prev, lam = 0, 0

    while True:
        ctr += 1
        lam = (x @ A @ x.T) / (x @ x.T)
        x = np.linalg.tensorsolve(A - lam[0][0] * E, x_0[0].T)
        x = np.reshape(x, (1, A.shape[0]))

        if np.abs(lam - lam_prev) < eps:
            print(f"На {ctr} итерации условие |lam - lam_prev| < eps перестает выполняться")
            break

        lam_prev = lam
        print(f"Итерация {ctr}:\n x_{ctr}:\n {x.T}")

    return ctr-1, x, lam_prev


if __name__ == "__main__":
    A = np.array([
        [12, 17, 8, 10, 10, 2],
        [15, 9, 3, 10, 4, 19],
        [10, 3, 10, 4, 15, 18],
        [13, 16, 0, 2, 0, 15],
        [9, 11, 5, 3, 1, 2],
        [9, 14, 17, 8, 15, 3]
    ])
    x_0 = np.array([[1, 1, 1, 1, 1, 1]])
    E = np.identity(A.shape[0])
    eps = 1e-6
    print("A:\n", A)
    print("E:\n", E)
    print("x_0:\n", x_0.T, "\n")

    ctr, x, lam = power_method()
    if ctr % 10 == 2:
        print(f"\nДля нашего набора данных потребовалась {ctr} итерация\n"
              f"Собственный вектор x_{ctr}:\n {x.T}\n"
              f"Максимальное по модулю собственное значение: {lam[0][0]}")
    elif 2 < ctr % 10 < 6:
        print(f"\nДля нашего набора данных потребовалось {ctr} итерации\n"
              f"Собственный вектор x_{ctr}:\n {x.T}\n"
              f"Максимальное по модулю собственное значение: {lam[0][0]}")
    else:
        print(f"\nДля нашего набора данных потребовалось {ctr} итераций\n"
              f"Собственный вектор x_{ctr}:\n {x.T}\n"
              f"Максимальное по модулю собственное значение: {lam[0][0]}")
