import numpy as np


def power_method():
    alpha = 0.01
    ctr = 1
    x_prev = x_0
    x_next = A @ x_prev / np.linalg.norm(A @ x_prev)
    print(f"Итерация {ctr}:\n x_{ctr}:\n {x_next}")
    lam_prev, lam = 0, 0
    x_prev = x_next

    while True:
        ctr += 1
        x_next = A @ x_prev / np.linalg.norm(A @ x_prev)
        lam = (x_next.T @ A @ x_next) / (x_next.T @ x_next)
        x_prev = x_prev + alpha * (x_next - x_prev)

        if np.abs(lam - lam_prev) > accelerated_eps_min:
            alpha += 0.035
        elif np.abs(lam - lam_prev) < accelerated_eps_max:
            alpha -= 0.035

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
    A = np.array([
        [12, 17, 8, 10, 10, 2],
        [15, 9, 3, 10, 4, 19],
        [10, 3, 10, 4, 15, 18],
        [13, 16, 0, 2, 0, 15],
        [9, 11, 5, 3, 1, 2],
        [9, 14, 17, 8, 15, 3]
    ])
    x_0 = np.array([[5, 1, 10, 5, 12, 6]]).T
    accelerated_eps_min = 1e-4
    accelerated_eps_max = 1e-1
    eps = 1e-6
    print("A:\n", A)
    print("x_0:\n", x_0, "\n")

    ctr, x, lam = power_method()
    if ctr % 10 == 2:
        print(f"\nДля нашего набора данных потребовалась {ctr} итерация\n"
              f"Собственный вектор x_{ctr}:\n {x}\n"
              f"Максимальное по модулю собственное значение: {lam[0][0]}")
    elif 2 < ctr % 10 < 6:
        print(f"\nДля нашего набора данных потребовалось {ctr} итерации\n"
              f"Собственный вектор x_{ctr}:\n {x}\n"
              f"Максимальное по модулю собственное значение: {lam[0][0]}")
    else:
        print(f"\nДля нашего набора данных потребовалось {ctr} итераций\n"
              f"Собственный вектор x_{ctr}:\n {x}\n"
              f"Максимальное по модулю собственное значение: {lam[0][0]}")
