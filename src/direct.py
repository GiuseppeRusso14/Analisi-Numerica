"""Metodi diretti per il problema ai minimi quadrati min_x 1/2 ||Ax - b||^2.

Equazioni normali: H x = c, con H = A^T A (simmetrica definita positiva se rank A = n)
e c = A^T b. Si risolvono con la fattorizzazione di Cholesky H = L L^T:
    L y = c       (sostituzione in avanti)
    L^T x = y     (sostituzione all'indietro)
"""

import numpy as np


class CholeskyError(np.linalg.LinAlgError):
    """Pivot non positivo: la matrice non è (numericamente) definita positiva."""


def forward_sub(L, b):
    """Risolve L y = b con L triangolare inferiore.

    y_i = (b_i - sum_{k<i} l_ik y_k) / l_ii,  i = 1..n.   Costo: circa n^2 flop.
    """
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def back_sub(U, y):
    """Risolve U x = y con U triangolare superiore.

    x_i = (y_i - sum_{k>i} u_ik x_k) / u_ii,  i = n..1.   Costo: circa n^2 flop.
    """
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def cholesky(H):
    """Fattorizzazione di Cholesky H = L L^T, per colonne.

    Per j = 1..n:
        l_jj = sqrt(h_jj - sum_{k<j} l_jk^2)
        l_ij = (h_ij - sum_{k<j} l_ik l_jk) / l_jj,   i > j
    Se l'argomento della radice è <= 0 la matrice non è definita positiva in
    aritmetica di macchina (ad esempio kappa(H) >~ 1/u): si solleva CholeskyError.
    Costo: circa n^3/3 flop.
    """
    n = H.shape[0]
    L = np.zeros_like(H, dtype=float)
    for j in range(n):
        d = H[j, j] - L[j, :j] @ L[j, :j]
        if d <= 0:
            raise CholeskyError(f"pivot non positivo alla colonna {j}: {d:.3e}")
        L[j, j] = np.sqrt(d)
        L[j + 1:, j] = (H[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
    return L


def solve_normal_cholesky(A, b):
    """Minimi quadrati con equazioni normali + Cholesky.

    Costo: formare H circa m n^2 flop, Cholesky n^3/3, sostituzioni 2 n^2.
    L'errore relativo su x cresce come u * kappa(A)^2, perché kappa(H) = kappa(A)^2.
    """
    H = A.T @ A
    c = A.T @ b
    L = cholesky(H)
    y = forward_sub(L, c)
    return back_sub(L.T, y)


if __name__ == "__main__":
    # Confronto con np.linalg.lstsq sul modello M2 (training 80%), dati grezzi e standardizzati
    from src.data import build_features, load_sgemm, standardize, target, train_test_split

    P, runs = load_sgemm()
    b = target(runs)
    itr, _ = train_test_split(len(b))
    A, _ = build_features(P, degree=2)
    A, b = A[itr], b[itr]

    for label, M in (("grezzo", A), ("standardizzato", standardize(A)[0])):
        x_ref = np.linalg.lstsq(M, b, rcond=None)[0]
        H = M.T @ M
        L = cholesky(H)
        x = solve_normal_cholesky(M, b)
        print(f"\nM2 {label}: kappa(A) = {np.linalg.cond(M):.2e}, kappa(H) = {np.linalg.cond(H):.2e}")
        print(f"  ||L L^T - H|| / ||H||      = {np.linalg.norm(L @ L.T - H) / np.linalg.norm(H):.2e}")
        print(f"  ||x - x_lstsq|| / ||x_lstsq|| = {np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref):.2e}")

    # Caso a rango deficiente: M2 con KWG^2 e KWI^2 (116 colonne, rango 114)
    A_full, _ = build_features(P[itr], degree=2, drop_dependent=False)
    print("\nM2 con KWG^2 e KWI^2 (rango deficiente):")
    try:
        x = solve_normal_cholesky(A_full, b)
        print(f"  Cholesky non si ferma, ma ||x|| = {np.linalg.norm(x):.2e}")
    except CholeskyError as e:
        print(f"  Cholesky fallisce: {e}")
