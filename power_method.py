import numpy as np


def error(x):
    return np.linalg.norm(x)


def power_method():
    ctr = 1
    x_prev = x_0
    x_next = A @ x_prev / np.linalg.norm(A @ x_prev)
    print(f"Итерация {ctr}:\n x_{ctr}:\n {x_next}\n ||x_{ctr}|| = {np.linalg.norm(x_next)}")
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
        print(f"Итерация {ctr}:\n x_{ctr}:\n {x_next}\n ||x_{ctr}|| = {np.linalg.norm(x_next)}")

    return ctr, x_prev, lam_prev


if __name__ == "__main__":
    # A = np.array([
    #     [0, 2],
    #     [2, 3]
    # ])
    # x_0 = np.array([[1, 1]]).T
    eps = 1e-3
    dim = 30
    A = np.random.randint(0, 10, (dim, dim))
    x_0 = np.random.randint(0, 10, (1, dim)).T
    print("A:\n", A)
    print("x_0:\n", x_0, "\n")

    ctr, x, lam = power_method()
    if ctr % 10 == 2:
        print(f"\nДля нашего набора данных потребовалась {ctr-1} итерация\n"
              f"Собственный вектор x_{ctr-1}:\n {x}\n"
              f"Максимальное собственное значение: {lam[0][0]}")
    elif 2 < ctr % 10 < 6:
        print(f"\nДля нашего набора данных потребовалось {ctr-1} итерации\n"
              f"Собственный вектор x_{ctr-1}:\n {x}\n"
              f"Максимальное собственное значение: {lam[0][0]}")
    else:
        print(f"\nДля нашего набора данных потребовалось {ctr-1} итераций\n"
              f"Собственный вектор x_{ctr-1}:\n {x}\n"
              f"Максимальное собственное значение: {lam[0][0]}")