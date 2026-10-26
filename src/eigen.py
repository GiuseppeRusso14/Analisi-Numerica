"""Stima degli autovalori estremi di H = A^T A (simmetrica definita positiva).

Servono a scegliere il passo dei metodi iterativi: Richardson converge se e solo se
0 < alpha < 2 / lambda_max, e il passo ottimo è alpha_opt = 2 / (lambda_max + lambda_min).
"""

import numpy as np

from src.direct import back_sub, cholesky, forward_sub


def power_method(H, tol=1e-10, maxit=10_000, seed=0):
    """Metodo delle potenze: stima di lambda_max e del suo autovettore.

    z^(k) = H q^(k-1),  q^(k) = z^(k) / ||z^(k)||,  nu^(k) = q^(k)^T H q^(k)  (Rayleigh).
    L'autovettore converge come |lambda_2 / lambda_1|^k; per H simmetrica il quoziente
    di Rayleigh converge come |lambda_2 / lambda_1|^(2k).
    Arresto: |nu^(k) - nu^(k-1)| <= tol * |nu^(k)|.
    Restituisce (nu, q, history) con history["nu"] = successione delle stime.
    """
    q = np.random.default_rng(seed).standard_normal(H.shape[0])
    q /= np.linalg.norm(q)
    Hq = H @ q
    nu = q @ Hq
    history = {"nu": [nu]}
    for _ in range(maxit):
        q = Hq / np.linalg.norm(Hq)
        Hq = H @ q
        nu_old, nu = nu, q @ Hq
        history["nu"].append(nu)
        if abs(nu - nu_old) <= tol * abs(nu):
            break
    return nu, q, history


def inverse_power_method(H, L=None, tol=1e-10, maxit=10_000, seed=0):
    """Metodo delle potenze inverse: stima di lambda_min e del suo autovettore.

    È il metodo delle potenze applicato a H^-1, il cui autovalore massimo è 1 / lambda_min.
    A ogni passo si risolve H z^(k) = q^(k-1) con il fattore di Cholesky H = L L^T,
    calcolato una volta sola (n^3/3 flop) e poi riusato: ogni passo costa solo due
    sostituzioni (2 n^2 flop). Si può passare L già calcolato.
    Velocità: |lambda_min / lambda_min,2|^k per l'autovettore (al quadrato per nu).
    Restituisce (nu, q, history).
    """
    if L is None:
        L = cholesky(H)
    q = np.random.default_rng(seed).standard_normal(H.shape[0])
    q /= np.linalg.norm(q)
    nu = q @ H @ q
    history = {"nu": [nu]}
    for _ in range(maxit):
        z = back_sub(L.T, forward_sub(L, q))  # z = H^-1 q
        q = z / np.linalg.norm(z)
        nu_old, nu = nu, q @ H @ q
        history["nu"].append(nu)
        if abs(nu - nu_old) <= tol * abs(nu):
            break
    return nu, q, history


def gershgorin_bounds(H):
    """Intervallo di Gershgorin che contiene tutti gli autovalori di H simmetrica.

    Ogni autovalore sta nell'unione dei dischi D_i = {z : |z - h_ii| <= R_i}, con
    R_i = sum_{j != i} |h_ij|. Per H simmetrica gli autovalori sono reali, quindi
    min_i (h_ii - R_i) <= lambda <= max_i (h_ii + R_i).
    Costo: n^2 operazioni, nessuna iterazione. Il bound inferiore è spesso negativo
    e quindi inutile per lambda_min.
    """
    d = np.diag(H)
    R = np.sum(np.abs(H), axis=1) - np.abs(d)
    return np.min(d - R), np.max(d + R)


if __name__ == "__main__":
    # Autovalori estremi di H = A^T A per il modello M2 e passi di Richardson che ne derivano
    from src.data import build_features, load_sgemm, standardize, target, train_test_split

    P, runs = load_sgemm()
    b = target(runs)
    itr, _ = train_test_split(len(b))
    A, _ = build_features(P, degree=2)
    A = A[itr]

    for label, M in (("grezzo", A), ("standardizzato", standardize(A)[0])):
        H = M.T @ M
        ev = np.linalg.eigvalsh(H)  # riferimento
        lmax, _, hmax = power_method(H)
        lmin, _, hmin = inverse_power_method(H)
        g_low, g_up = gershgorin_bounds(H)
        print(f"\nM2 {label}:  (riferimento np.linalg.eigvalsh)")
        print(f"  potenze:          lambda_max = {lmax:.6e}  err. rel. {abs(lmax - ev[-1]) / ev[-1]:.1e}"
              f"  in {len(hmax['nu']) - 1} it.   (lambda_2/lambda_1 = {ev[-2] / ev[-1]:.4f})")
        print(f"  potenze inverse:  lambda_min = {lmin:.6e}  err. rel. {abs(lmin - ev[0]) / ev[0]:.1e}"
              f"  in {len(hmin['nu']) - 1} it.   (lambda_min/lambda_min,2 = {ev[0] / ev[1]:.4f})")
        print(f"  Gershgorin:       [{g_low:.3e}, {g_up:.3e}]   (g_up / lambda_max = {g_up / ev[-1]:.2f})")
        print(f"  kappa(H) stimato = {lmax / lmin:.3e}   (vero {ev[-1] / ev[0]:.3e})")
        print(f"  passi Richardson: 2/lambda_max = {2 / lmax:.3e}   alpha_opt = {2 / (lmax + lmin):.3e}"
              f"   Gershgorin: 2/g_up = {2 / g_up:.3e}")
