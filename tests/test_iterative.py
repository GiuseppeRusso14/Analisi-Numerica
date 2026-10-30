"""Test dei metodi iterativi su problemi sintetici con soluzione e
condizionamento noti (kappa(A) = 10, quindi kappa(H) = 100)."""

import numpy as np

from src.data import make_synthetic
from src.iterative import bfgs, conjugate_gradient, richardson, steepest_descent

KAPPA_A = 10.0
KAPPA_H = KAPPA_A**2
RHO = (KAPPA_H - 1) / (KAPPA_H + 1)          # fattore di riduzione teorico (F6, F7)
LMAX, LMIN = 1.0, 1.0 / KAPPA_H              # valori singolari tra 1 e 1/kappa(A)


def rel(u, v):
    return np.linalg.norm(u - v) / np.linalg.norm(v)


def test_richardson_converge():
    A, b, x_true = make_synthetic(300, 20, KAPPA_A, residual=0.5)
    x, h = richardson(A, b, 2 / (LMAX + LMIN), tol=1e-12, maxit=20_000, x_star=x_true)
    assert rel(x, x_true) < 1e-9
    assert h["grad_norm"][-1] <= 1e-12 * h["grad_norm"][0]


def test_richardson_velocita_teorica():
    # con alpha_opt l'errore si riduce al piu' di rho_opt per iterazione: ||e_k|| <= rho^k ||e_0||
    A, b, x_true = make_synthetic(300, 20, KAPPA_A)
    _, h = richardson(A, b, 2 / (LMAX + LMIN), tol=0, maxit=500, x_star=x_true)
    err = np.array(h["err"])
    k = np.arange(len(err))
    assert np.all(err <= RHO**k * err[0] * (1 + 1e-10))
    # e asintoticamente il fattore e' proprio rho (componenti lungo lambda_max e lambda_min)
    assert abs(err[500] / err[499] - RHO) < 1e-3


def test_richardson_diverge_oltre_2_su_lambda_max():
    A, b, _ = make_synthetic(300, 20, KAPPA_A)
    _, h = richardson(A, b, 1.1 * 2 / LMAX, tol=0, maxit=200)
    assert h["grad_norm"][-1] > 1e3 * h["grad_norm"][0]


def test_steepest_descent_converge():
    A, b, x_true = make_synthetic(300, 20, KAPPA_A, residual=0.5)
    x, _ = steepest_descent(A, b, tol=1e-12, maxit=20_000, x_star=x_true)
    assert rel(x, x_true) < 1e-9


def test_steepest_descent_proprieta():
    A, b, x_true = make_synthetic(300, 20, KAPPA_A)
    H = A.T @ A
    _, h = steepest_descent(A, b, tol=0, maxit=200)
    # f decresce a ogni passo (passo esatto lungo una direzione di discesa)
    assert np.all(np.diff(h["f"]) < 0)
    # iterate x_0, ..., x_40 (con maxit=k il metodo si ferma all'iterata k)
    xs = [steepest_descent(A, b, tol=0, maxit=k)[0] for k in range(41)]
    rs = [A.T @ (b - A @ x) for x in xs]
    e_H = [np.sqrt((x - x_true) @ H @ (x - x_true)) for x in xs]
    for k in range(1, 41):
        # gradienti consecutivi ortogonali (zig-zag)
        assert abs(rs[k] @ rs[k - 1]) < 1e-10 * np.linalg.norm(rs[k]) * np.linalg.norm(rs[k - 1])
        # stima in norma dell'energia: ||e_k||_H <= rho^k ||e_0||_H
        assert e_H[k] <= RHO**k * e_H[0] * (1 + 1e-10)


# --- gradiente coniugato e BFGS ---------------------------------------------------

def test_cg_termina_in_n_passi():
    # in aritmetica esatta CG termina in al piu' n passi; con kappa piccolo si vede anche in macchina
    A, b, x_true = make_synthetic(200, 15, 3.0, residual=0.3)
    x, h = conjugate_gradient(A, b, tol=0, maxit=15)
    assert len(h["grad_norm"]) - 1 == 15
    assert rel(x, x_true) < 1e-10


def test_cg_stima_di_convergenza():
    # ||e_k||_H <= 2 ((sqrt(kappa)-1)/(sqrt(kappa)+1))^k ||e_0||_H, kappa = kappa(H) = 100
    A, b, x_true = make_synthetic(300, 30, KAPPA_A)
    H = A.T @ A
    q = (np.sqrt(KAPPA_H) - 1) / (np.sqrt(KAPPA_H) + 1)
    e_H0 = np.sqrt(x_true @ H @ x_true)                     # x_0 = 0
    for k in range(1, 30):
        e = conjugate_gradient(A, b, tol=0, maxit=k)[0] - x_true
        assert np.sqrt(e @ H @ e) <= 2 * q**k * e_H0 * (1 + 1e-10)


def test_cg_piu_veloce_di_steepest_descent():
    A, b, _ = make_synthetic(300, 30, KAPPA_A)
    _, h_cg = conjugate_gradient(A, b, tol=1e-10, maxit=1_000)
    _, h_sd = steepest_descent(A, b, tol=1e-10, maxit=10_000)
    assert len(h_cg["grad_norm"]) < len(h_sd["grad_norm"]) / 5


def test_bfgs_uguale_a_cg():
    # su una quadratica, con H_0 = I e passo esatto, BFGS genera le stesse iterate di CG
    A, b, x_true = make_synthetic(200, 15, 3.0, residual=0.3)
    for k in range(1, 16):
        x_cg = conjugate_gradient(A, b, tol=0, maxit=k)[0]
        x_bfgs = bfgs(A, b, tol=0, maxit=k)[0]
        assert np.linalg.norm(x_bfgs - x_cg) < 1e-8 * np.linalg.norm(x_true)


def test_bfgs_converge():
    A, b, x_true = make_synthetic(300, 30, KAPPA_A, residual=0.5)
    x, h = bfgs(A, b, tol=1e-12, maxit=200)
    assert rel(x, x_true) < 1e-9
    assert np.all(np.diff(h["f"]) <= 1e-12 * h["f"][0])      # f non cresce
