# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

R = 8
#      np.log(X+1)**2 + np.log(Y+1)**2 <= (R / 4)**2
# <--> (4*np.log(X+1))**2 + (4*np.log(Y+1))**2 <= R**2
# <--> (np.log((X+1)**4))**2 + (np.log((Y+1)**4))**2 <= R**2
d = (R+1)**4

xs = np.linspace(0, d, 100)
X, Y = np.meshgrid(xs, xs)
X, Y = np.meshgrid(xs, xs)
Z = np.log(X+1)**2 + np.log(Y+1)**2 <= R**2

sg = []
for x in range(R**4):
    for y in range(R**4):
        if np.log(x+1)**2 + np.log(y+1)**2 <= R**2:
            sg.append((x, y))
        else:
            break

M = np.zeros((d, d))
for index in sg:
    M[index] = 1
assert np.all(M == M.T)
es, vs = np.linalg.eigh(M)
rank = np.count_nonzero(abs(es) > 1e-12 * np.linalg.norm(es))

sg = np.array(sg)
plt.contourf(X, Y, Z)
plt.scatter(sg[:, 0], sg[:, 1], s=1, c="tab:red")
plt.show()


# I now want to find a natural number K such that:
#     (x, y) in sg  <-->  x <= K or y <= K
# M has the property, that
#     M[i, j] == 0   -->  M[i:, j:] == 0
# This means, that K is just the number of non-zero diagonal elements in M.
# I can hence write M (roughly) as the sum
#     M = M[:K] + M[K:, :K],
# which means that rank(M) <= rank(M[:K]) + rank(M[K:, :K]) <= 2*K,
# where the last inequality follows trivially because M[:K] has K rows and M[K:, :K] has K columns.

K = np.count_nonzero(np.diag(M))
assert np.all(M[K:, K:] == 0)
assert rank <= 2*K


# Counting the non-zero elements of M can also be done algebraically,
# without the need to ever construct M explicitly.
# Recall that 
#     M[i, i] != 0  <-->  (i, i) in sg
#                   <-->  2 * ln(i+1)**2 <= R**2
#                   <-->  ln(i+1) <= sqrt(R**2 / 2) == R / sqrt(2)
#                   <-->  i <= exp(R / sqrt(2)) - 1
# This means that the diagonal indeces have to satisfy the bound i <= exp(R / sqrt(2)) - 1 in order to have a non-zero entry.
# The number of such indices is given by
#     K = floor(exp(R / sqrt(2)) - 1) + 1 == floor(exp(R / sqrt(2))) .

assert K == np.floor(np.exp(R / np.sqrt(2)))

size = len(str(d**2))
print(f"Dimension:  {d**2:{size}d}")
print(f"Terms:      {len(sg):{size}d}")
print(f"Rank:       {rank:{size}d}")
print(f"Rank bound: {2*K:{size}d}")
