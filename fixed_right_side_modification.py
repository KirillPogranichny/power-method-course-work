import numpy as np
from datetime import datetime


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
    start_time = datetime.now()
    SEED = 1
    dim = 3
    np.random.seed(SEED)
    eps = 1e-6

    A = np.random.randint(0, 100, (dim, dim))
    x_0 = np.random.randint(0, 100, (1, dim))
    E = np.identity(dim)
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

    print(f"\nВремя работы алгоритма: {datetime.now() - start_time}")
