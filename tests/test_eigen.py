"""Test del metodo delle potenze, delle potenze inverse e di Gershgorin
su matrici simmetriche definite positive con autovalori noti."""

import numpy as np

from src.direct import cholesky
from src.eigen import gershgorin_bounds, inverse_power_method, power_method


def spd_con_autovalori(lam, seed=0):
    """H = Q diag(lam) Q^T con Q ortogonale casuale: autovalori esattamente lam."""
    n = len(lam)
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))
    return Q @ np.diag(lam) @ Q.T


LAM = np.array([10.0, 5.0] + list(np.linspace(1.0, 4.0, 18)) + [0.5])  # 21 autovalori


def test_potenze_lambda_max():
    H = spd_con_autovalori(LAM)
    nu, q, _ = power_method(H, tol=1e-14)
    assert abs(nu - 10.0) / 10.0 < 1e-6
    assert np.linalg.norm(H @ q - nu * q) < 1e-5          # q è (quasi) un autovettore


def test_potenze_inverse_lambda_min():
    H = spd_con_autovalori(LAM)
    nu, q, _ = inverse_power_method(H, tol=1e-14)
    assert abs(nu - 0.5) / 0.5 < 1e-6
    # riuso del fattore di Cholesky: stesso risultato passando L
    nu_L, _, _ = inverse_power_method(H, L=cholesky(H), tol=1e-14)
    assert nu_L == nu


def test_rayleigh_per_difetto():
    # il quoziente di Rayleigh di H simmetrica sta sempre in [lambda_min, lambda_max]:
    # le stime di lambda_max arrivano dal basso, quelle di lambda_min dall'alto
    H = spd_con_autovalori(LAM)
    _, _, h_max = power_method(H, tol=1e-14)
    _, _, h_min = inverse_power_method(H, tol=1e-14)
    assert np.all(np.array(h_max["nu"]) <= 10.0 + 1e-12)
    assert np.all(np.array(h_min["nu"]) >= 0.5 - 1e-12)


def test_velocita_potenze():
    # errore sul quoziente di Rayleigh ~ C (lambda_2 / lambda_1)^(2k) = C 0.25^k.
    # Il rapporto vale 0.25 solo asintoticamente (all'inizio pesa anche lambda_3 = 4):
    # si guardano le iterazioni 12-20, prima che l'errore arrivi agli arrotondamenti
    H = spd_con_autovalori(LAM)
    _, _, h = power_method(H, tol=0, maxit=20)
    err = 10.0 - np.array(h["nu"])
    rapporti = err[13:20] / err[12:19]
    assert np.allclose(rapporti, 0.25, rtol=0.01)


def test_gershgorin_contiene_lo_spettro():
    H = spd_con_autovalori(LAM)
    low, up = gershgorin_bounds(H)
    assert low <= 0.5 and up >= 10.0
    D = np.diag([3.0, 7.0, 1.0])                              # matrice diagonale: dischi = punti
    assert gershgorin_bounds(D) == (1.0, 7.0)
