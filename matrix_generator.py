import numpy as np


SEED = 220144
np.random.seed(SEED)
dim = 10

while True:
    A = np.random.randint(0, 100, (dim, dim))
    # x_0 = np.random.randint(0, 20, (1, dim)).T
    if any(np.iscomplex(np.linalg.eigvals(A))):
        SEED += 1
        print(f"SEED = {SEED}")
    else:
        break

print("A:\n", A)
# print("\nx_0:\n", x_0, "\n")

print(f"λ = {np.linalg.eigvals(A)}")
