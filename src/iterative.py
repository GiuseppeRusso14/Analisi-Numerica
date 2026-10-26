"""Metodi iterativi per min_x f(x) = 1/2 ||Ax - b||^2.

Gradiente: grad f(x) = A^T (Ax - b) = Hx - c;  residuo r = c - Hx = -grad f(x).
H = A^T A non si forma mai: ogni prodotto H p si calcola come A^T (A p), con costo
O(mn) per iterazione invece di mn^2 una volta sola.

Ogni metodo restituisce (x, history), con history = dict di liste, una voce per iterata:
    grad_norm  ||grad f(x^(k))||
    step_norm  ||x^(k) - x^(k-1)||      (0 per k = 0)
    f          f(x^(k))
    err        ||x^(k) - x*||           (solo se x_star è dato)
    time       secondi dall'inizio
Arresto: ||grad f(x^(k))|| <= tol * ||grad f(x^(0))||, oppure maxit iterazioni.
"""

from time import perf_counter

import numpy as np


def _new_history(x_star):
    keys = ["grad_norm", "step_norm", "f", "time"] + (["err"] if x_star is not None else [])
    return {k: [] for k in keys}


def _record(history, x, s, r, step, t0, x_star):
    """Salva i dati dell'iterata x, con s = b - Ax (residuo dei dati) e r = A^T s."""
    history["grad_norm"].append(np.linalg.norm(r))
    history["step_norm"].append(step)
    history["f"].append(0.5 * (s @ s))
    history["time"].append(perf_counter() - t0)
    if x_star is not None:
        history["err"].append(np.linalg.norm(x - x_star))


def richardson(A, b, alpha, x0=None, tol=1e-8, maxit=10_000, x_star=None):
    """Richardson stazionario = metodo del gradiente a passo fisso.

    x^(k+1) = x^(k) + alpha r^(k),   r^(k) = c - H x^(k) = -grad f(x^(k)).
    Matrice d'iterazione B = I - alpha H: converge se e solo se 0 < alpha < 2 / lambda_max.
    Con alpha_opt = 2 / (lambda_max + lambda_min):  ||e^(k)|| <= ((kappa-1)/(kappa+1))^k ||e^(0)||,
    con kappa = kappa(H). Costo per iterazione: due prodotti con A (4mn flop).
    """
    t0 = perf_counter()
    x = np.zeros(A.shape[1]) if x0 is None else np.array(x0, dtype=float)
    history = _new_history(x_star)
    s = b - A @ x
    r = A.T @ s
    _record(history, x, s, r, 0.0, t0, x_star)
    g0 = history["grad_norm"][0]
    for _ in range(maxit):
        if history["grad_norm"][-1] <= tol * g0:
            break
        dx = alpha * r
        x = x + dx
        s = b - A @ x
        r = A.T @ s
        _record(history, x, s, r, np.linalg.norm(dx), t0, x_star)
        if not np.isfinite(history["grad_norm"][-1]):
            break  # divergenza (alpha troppo grande)
    return x, history


def steepest_descent(A, b, x0=None, tol=1e-8, maxit=10_000, x_star=None):
    """Metodo del gradiente con passo ottimo (steepest descent).

    Direzione p^(k) = r^(k) = -grad f(x^(k)); passo esatto (minimo di f lungo p):
        alpha_k = r^T r / (r^T H r) = ||r||^2 / ||A r||^2.
    Stima: ||e^(k)||_H <= ((kappa-1)/(kappa+1))^k ||e^(0)||_H, con ||e||_H = sqrt(e^T H e).
    Il residuo dei dati si aggiorna, s^(k+1) = s^(k) - alpha_k A r^(k), così ogni iterazione
    costa due prodotti con A, come Richardson.
    """
    t0 = perf_counter()
    x = np.zeros(A.shape[1]) if x0 is None else np.array(x0, dtype=float)
    history = _new_history(x_star)
    s = b - A @ x
    r = A.T @ s
    _record(history, x, s, r, 0.0, t0, x_star)
    g0 = history["grad_norm"][0]
    for _ in range(maxit):
        if history["grad_norm"][-1] <= tol * g0:
            break
        Ar = A @ r
        alpha = (r @ r) / (Ar @ Ar)
        dx = alpha * r
        x = x + dx
        s = s - alpha * Ar
        r = A.T @ s
        _record(history, x, s, r, np.linalg.norm(dx), t0, x_star)
    return x, history


if __name__ == "__main__":
    # Richardson e steepest descent sul modello M2 (training 80%), primi passi.
    # Ogni iterazione costa ~10 ms (due prodotti con A 193280 x 114): qui ci si ferma a
    # poche migliaia di iterazioni; le esecuzioni complete sono nell'esperimento E2.
    from src.data import build_features, load_sgemm, standardize, target, train_test_split
    from src.direct import solve_svd
    from src.eigen import gershgorin_bounds, inverse_power_method, power_method

    P, runs = load_sgemm()
    b = target(runs)
    itr, _ = train_test_split(len(b))
    A, _ = build_features(P, degree=2)
    A, b = A[itr], b[itr]

    for label, M, maxit in (("standardizzato", standardize(A)[0], 2_000), ("grezzo", A, 500)):
        H = M.T @ M
        lmax, _, _ = power_method(H)
        lmin, _, _ = inverse_power_method(H)
        _, g_up = gershgorin_bounds(H)
        kappa = lmax / lmin
        x_star = solve_svd(M, b)
        print(f"\nM2 {label}: kappa(H) = {kappa:.2e}, rho_opt = {(kappa - 1) / (kappa + 1):.6f}, {maxit} iterazioni")
        for name, run in (
            ("Richardson alpha_opt", lambda: richardson(M, b, 2 / (lmax + lmin), tol=0, maxit=maxit, x_star=x_star)),
            ("Richardson Gershgorin", lambda: richardson(M, b, 2 / g_up, tol=0, maxit=maxit, x_star=x_star)),
            ("steepest descent", lambda: steepest_descent(M, b, tol=0, maxit=maxit, x_star=x_star)),
        ):
            _, h = run()
            g, e = h["grad_norm"], h["err"]
            print(f"  {name:22s} ||grad||/||grad0|| = {g[-1] / g[0]:.1e}   ||e||/||e0|| = {e[-1] / e[0]:.1e}"
                  f"   tempo {h['time'][-1]:.1f} s")

    # alpha appena oltre 2 / lambda_max: divergenza
    M = standardize(A)[0]
    lmax, _, _ = power_method(M.T @ M)
    _, h = richardson(M, b, 1.05 * 2 / lmax, maxit=300)
    print(f"\nRichardson con alpha = 1.05 * 2/lambda_max: ||grad|| da {h['grad_norm'][0]:.1e}"
          f" a {h['grad_norm'][-1]:.1e} in 300 iterazioni")
