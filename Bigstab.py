import numpy as np

def Bigstab(b, A, x, r_0, p):
    rk = r_0
    rk_1 = r_0
    iterr = 0
    while((max(rk.min(), rk.max(), key = abs) / max(b.min(), b.max(), key = abs)) > 0.00001):
        v = A.dot(p)
        ak = np.dot(r_0, rk_1) / np.dot(v, rk_1)
        x = x + ak * p
        rk = rk - ak * v
        v = A.dot(v)
        bk = np.dot(r_0, v) / np.dot(r_0, rk_1)
        p = rk + bk * (p - ak * v)
        rk_1 = (A.transpose()).dot(rk) - bk * (A.transpose()).dot(v)
        iterr += 1
    return x, iterr

b = np.array([4.2108, 4.6174, -5.8770, 2.7842, 0.2178])
A = np.array([[0.6897, -0.0908, 0.0182, 0.0363, 0.1271],
    [0.0944, 1.0799, 0, -0.0726, 0.0726],
    [0.0545, 0, 0.8676, -0.2541, 0.1452],
    [-0.1089, 0.2287, 0, 0.8531, -0.0363],
    [0.4538, 0, 0.1634, 0.0182, 1.0164]])
x_0 = np.zeros(5)
r_0 = b - A.dot(x_0)
p = r_0
print(Bigstab(b, A, x_0, r_0, p))