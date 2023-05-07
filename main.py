import numpy as np


def initial(dim):
	A = np.random.randint(0, 10, (dim, dim))
	b = np.random.randint(0, 10, (1, dim)).T
	return A, b


def power_method(A, b, eps, max_iter, lam_prev):
	for i in range(max_iter):
		b = np.matmul(A, b) / np.linalg.norm(np.matmul(A, b))
		lam = np.matmul(np.matmul(b.T, A), b) / np.matmul(b.T, b)
		if np.abs(lam - lam_prev) < eps:
			break
		lam_prev = lam
	return lam, b


if __name__ == "__main__":
	np.random.seed(20)
	eps = 1e-6
	max_iter = 100
	lam_prev = 0
	dim = int(input("Enter dimension of matrix: "))

	A, b = initial(dim)
	print("\nInitial data\nA:\n", A)
	print("b:\n", b)

	lam, b = power_method(A, b, eps, max_iter, lam_prev)
	print("\nSolution\nLargest eigenvalue:", float(lam))
	print("b:\n", b)
