import numpy as np
from datetime import datetime


def power_method():
    ctr = 1
    x_prev = x_0
    x_next = A @ x_prev / np.linalg.norm(A @ x_prev)
    print(f"Итерация {ctr}:\n x_{ctr}:\n {x_next}")
    lam_prev, lam = 0, 0

    while True:
        ctr += 1
        x_prev = x_next
        x_next = A @ x_prev / np.linalg.norm(A @ x_prev)
        lam = (x_next.T @ A @ x_next) / (x_next.T @ x_next)

        if np.abs(lam - lam_prev) < eps:
            print(f"На {ctr} итерации условие |lam - lam_prev| < eps перестает выполняться")
            break
        elif np.linalg.norm(x_next - x_prev) < eps:
            print(f"На {ctr} итерации условие ||x_next - x_prev|| < eps перестает выполняться")
            break

        lam_prev = lam
        print(f"Итерация {ctr}:\n x_{ctr}:\n {x_next}")

    return ctr-1, x_prev, lam_prev


if __name__ == "__main__":
    start_time = datetime.now()
    SEED = 1
    dim = 3
    np.random.seed(SEED)
    eps = 1e-6

    A = np.array([
        [37, 12, 72],
        [9, 75, 5],
        [79, 64, 16]
    ])
    x_0 = np.random.randint(0, 100, (1, dim)).T
    print("A:\n", A)
    print("x_0:\n", x_0, "\n")

    ctr, x, lam = power_method()
    if ctr % 10 == 1:
        print(f"\nДля нашего набора данных потребовалась {ctr} итерация\n"
              f"Собственный вектор x_{ctr}:\n {x}\n"
              f"Максимальное по модулю собственное значение: {lam[0][0]}")
    elif 1 < ctr % 10 < 5:
        print(f"\nДля нашего набора данных потребовалось {ctr} итерации\n"
              f"Собственный вектор x_{ctr}:\n {x}\n"
              f"Максимальное по модулю собственное значение: {lam[0][0]}")
    else:
        print(f"\nДля нашего набора данных потребовалось {ctr} итераций\n"
              f"Собственный вектор x_{ctr}:\n {x}\n"
              f"Максимальное по модулю собственное значение: {lam[0][0]}")

    print(f"Время работы алгоритма: {datetime.now() - start_time}")
