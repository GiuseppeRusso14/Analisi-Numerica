"""Metodi diretti per il problema ai minimi quadrati min_x 1/2 ||Ax - b||^2.

Equazioni normali: H x = c, con H = A^T A (simmetrica definita positiva se rank A = n)
e c = A^T b. Si risolvono con la fattorizzazione di Cholesky H = L L^T:
    L y = c       (sostituzione in avanti)
    L^T x = y     (sostituzione all'indietro)

QR di Householder e SVD lavorano invece direttamente su A, senza formare H.
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



def householder_qr(A):
    """Fattorizzazione QR di Householder: A = Q R, con R = [R1; 0].

    Alla colonna k si sceglie la riflessione P_k = I - 2 v v^T (con ||v|| = 1) che manda
    a = A[k:, k] in un multiplo di e_1:
        v = a + sign(a_1) ||a|| e_1,   poi normalizzato.
    Il segno evita la cancellazione tra a_1 e ||a||. Q non si forma: si restituiscono
    R1 (n x n) e la lista dei vettori v_k, da applicare a b con apply_qt.
    Costo: circa 2 m n^2 - (2/3) n^3 flop.
    """
    W = np.array(A, dtype=float)
    m, n = W.shape
    V = []
    for k in range(n):
        a = W[k:, k]
        norm_a = np.linalg.norm(a)
        v = a.copy()
        v[0] += (1.0 if a[0] >= 0 else -1.0) * norm_a
        norm_v = np.linalg.norm(v)
        if norm_v > 0:
            v /= norm_v
            W[k:, k:] -= 2.0 * np.outer(v, v @ W[k:, k:])
        V.append(v)
    return np.triu(W[:n]), V


def apply_qt(V, b):
    """Calcola Q^T b applicando in ordine le riflessioni P_1, ..., P_n."""
    y = np.array(b, dtype=float)
    for k, v in enumerate(V):
        y[k:] -= 2.0 * v * (v @ y[k:])
    return y


def solve_qr(A, b):
    """Minimi quadrati con QR di Householder: R1 x = (Q^T b)_{1:n}.

    ||Ax - b|| = ||Q^T (Ax - b)||, e Q^T (Ax - b) = [R1 x - (Q^T b)_{1:n}; -(Q^T b)_{n+1:m}]:
    la prima parte si annulla risolvendo il sistema triangolare, la seconda è il residuo
    minimo. L'errore relativo su x cresce come u * kappa(A) se il residuo è piccolo.
    """
    n = A.shape[1]
    R, V = householder_qr(A)
    y = apply_qt(V, b)
    return back_sub(R, y[:n])


def solve_svd(A, b):
    """Minimi quadrati con la SVD ridotta A = U S V^T: x = sum_i (u_i^T b / s_i) v_i.

    Versione troncata: si tengono solo i valori singolari s_i > tau, con
    tau = max(m, n) * eps * s_1. Se A ha rango deficiente si ottiene la soluzione
    di norma minima.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    tau = max(A.shape) * np.finfo(float).eps * s[0]
    keep = s > tau
    return Vt[keep].T @ ((U[:, keep].T @ b) / s[keep])


if __name__ == "__main__":
    # Confronto dei tre metodi diretti sul modello M2 (training 80%), con np.linalg.lstsq
    from time import perf_counter

    from src.data import build_features, load_sgemm, standardize, target, train_test_split

    def rel(u, v):
        return np.linalg.norm(u - v) / np.linalg.norm(v)

    P, runs = load_sgemm()
    b = target(runs)
    itr, _ = train_test_split(len(b))
    A, _ = build_features(P, degree=2)
    A, b = A[itr], b[itr]
    solvers = (("Cholesky", solve_normal_cholesky), ("QR", solve_qr), ("SVD", solve_svd))

    for label, M in (("grezzo", A), ("standardizzato", standardize(A)[0])):
        x_ref = np.linalg.lstsq(M, b, rcond=None)[0]
        print(f"\nM2 {label}: kappa(A) = {np.linalg.cond(M):.2e}")
        for name, solve in solvers:
            t0 = perf_counter()
            x = solve(M, b)
            t = perf_counter() - t0
            print(f"  {name:8s}  ||x - x_lstsq|| / ||x_lstsq|| = {rel(x, x_ref):.2e}   tempo {t:5.2f} s")

    # Q ortogonale conserva la norma di Frobenius: ||R1||_F = ||A||_F
    R, V = householder_qr(A)
    print(f"\nQR grezzo: ||R1||_F / ||A||_F - 1 = {np.linalg.norm(R) / np.linalg.norm(A) - 1:.1e}")

    # Caso a rango deficiente: M2 con KWG^2 e KWI^2 (116 colonne, rango 114)
    A_full, _ = build_features(P[itr], degree=2, drop_dependent=False)
    x_svd = solve_svd(A_full, b)
    print("\nM2 con KWG^2 e KWI^2 (rango deficiente):")
    try:
        solve_normal_cholesky(A_full, b)
    except CholeskyError as e:
        print(f"  Cholesky fallisce: {e}")
    R_full, _ = householder_qr(A_full)
    d = np.abs(np.diag(R_full))
    x_qr = solve_qr(A_full, b)
    print(f"  QR: |r_kk| min / max = {d.min() / d.max():.1e}   ||x|| = {np.linalg.norm(x_qr):.2e}")
    print(f"  SVD: ||x|| = {np.linalg.norm(x_svd):.2e}   "
          f"predizioni uguali a M2 senza i due quadrati: {rel(A_full @ x_svd, A @ np.linalg.lstsq(A, b, rcond=None)[0]):.1e}")
