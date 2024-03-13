import numpy as np
import matplotlib.pyplot as plt


def power_iteration(A, tol=0.000001):
    truth = np.linalg.eig(A)
    print(truth.eigenvectors)
    v = np.random.normal(size=A.shape[1])
    end = np.empty(shape=A.shape[1])
    v = v / np.linalg.norm(v)
    previous = np.empty(shape=A.shape[1])
    while True:
        previous[:] = v
        v = A @ v
        v = v / np.linalg.norm(v)
        visualise(v, truth.eigenvectors)
        if np.allclose(v, previous, atol=tol):
            end[:] = v
            break
    return v


def visualise(calculated, truth):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if truth.ndim != calculated.ndim:
        calculated = calculated[np.newaxis, :]
    uvw = np.concatenate([calculated, truth], axis=0)
    u = uvw[:, 0]
    v = uvw[:, 1]
    w = uvw[:, 2]
    origin = np.zeros_like(u)
    ax.quiver(origin, origin, origin, u, v, w, normalize=True)
    max_abs = max(np.max(np.abs(calculated)), np.max(np.abs(truth)))
    ax.set_xlim3d(-max_abs, max_abs)
    ax.set_ylim3d(-max_abs, max_abs)
    ax.set_zlim3d(-max_abs, max_abs)
    plt.show()
    input("Press Enter to continue...")


def simultaneous_orthogonalisation(A, tol=0.000001):
    truth = np.linalg.eig(A)
    Q, R = np.linalg.qr(A)
    previous = np.empty(shape=Q.shape)
    for i in range(100):
        previous[:] = Q
        X = A @ Q
        Q, R = np.linalg.qr(X)
        visualise(Q, truth.eigenvectors)
        if np.allclose(Q, previous, atol=tol):
            break
    return Q


def qr_algorithm(A, tol=0.0001):
    Q, R = np.linalg.qr(A)
    previous = np.empty(shape=Q.shape)
    for i in range(500):
        previous[:] = Q
        X = R @ Q
        Q, R = np.linalg.qr(X)
        if np.allclose(X, np.triu(X), atol=tol):
            break
    return Q


A = np.random.rand(3, 3)
A = A @ A.T
print(f"a is {A}")
v = simultaneous_orthogonalisation(A)
print(v)
value = A @ v
print(value)
eigen = value / v
print(eigen)
