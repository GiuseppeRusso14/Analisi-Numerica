"""Test dei metodi diretti: correttezza su problemi sintetici con soluzione nota
e comportamento sul caso a rango deficiente del dataset reale."""

import numpy as np
import pytest

from src.data import build_features, load_sgemm, make_synthetic, target
from src.direct import (
    CholeskyError,
    apply_qt,
    back_sub,
    cholesky,
    forward_sub,
    householder_qr,
    solve_normal_cholesky,
    solve_qr,
    solve_svd,
)

EPS = np.finfo(float).eps
SOLVERS = [solve_normal_cholesky, solve_qr, solve_svd]


def rel(u, v):
    return np.linalg.norm(u - v) / np.linalg.norm(v)


# --- sostituzioni ---------------------------------------------------------------

def test_sostituzioni():
    rng = np.random.default_rng(0)
    n = 30
    L = np.tril(rng.standard_normal((n, n))) + n * np.eye(n)  # diagonale dominante
    b = rng.standard_normal(n)
    assert rel(forward_sub(L, b), np.linalg.solve(L, b)) < 1e-13
    assert rel(back_sub(L.T, b), np.linalg.solve(L.T, b)) < 1e-13


# --- Cholesky -------------------------------------------------------------------

def test_cholesky_fattorizzazione():
    A, _, _ = make_synthetic(200, 20, kappa=1e3)
    H = A.T @ A
    L = cholesky(H)
    assert np.allclose(L, np.tril(L))          # triangolare inferiore
    assert np.all(np.diag(L) > 0)              # diagonale positiva
    assert np.linalg.norm(L @ L.T - H) / np.linalg.norm(H) < 10 * EPS


def test_cholesky_matrice_non_definita_positiva():
    H = np.array([[1.0, 2.0], [2.0, 1.0]])     # autovalori 3 e -1
    with pytest.raises(CholeskyError):
        cholesky(H)


# --- QR di Householder ------------------------------------------------------------

def test_householder_qr():
    m, n = 60, 10
    A, _, _ = make_synthetic(m, n, kappa=1e2)
    R, V = householder_qr(A)
    QT = np.column_stack([apply_qt(V, e) for e in np.eye(m)])  # Q^T, formata solo qui
    assert np.linalg.norm(QT @ QT.T - np.eye(m)) < 100 * EPS  # Q ortogonale
    assert np.allclose(R, np.triu(R))
    QTA = QT @ A
    assert np.linalg.norm(QTA[:n] - R) / np.linalg.norm(A) < 100 * EPS
    assert np.linalg.norm(QTA[n:]) / np.linalg.norm(A) < 100 * EPS


# --- solutori dei minimi quadrati ------------------------------------------------

@pytest.mark.parametrize("solve", SOLVERS)
@pytest.mark.parametrize("residual", [0.0, 0.5])
def test_soluzione_ben_condizionato(solve, residual):
    A, b, x_true = make_synthetic(500, 30, kappa=10, residual=residual)
    assert rel(solve(A, b), x_true) < 1e-12
    assert rel(solve(A, b), np.linalg.lstsq(A, b, rcond=None)[0]) < 1e-12


@pytest.mark.parametrize("solve", SOLVERS)
def test_residuo_ortogonale(solve):
    # condizione di ottimalita': A^T (b - A x) = 0
    A, b, _ = make_synthetic(500, 30, kappa=100, residual=1.0)
    x = solve(A, b)
    assert np.linalg.norm(A.T @ (b - A @ x)) / (np.linalg.norm(A) * np.linalg.norm(b)) < 1e-13


def test_cholesky_perde_piu_cifre_di_qr():
    # residuo nullo, kappa(A) = 1e6: errore ~ u kappa^2 per Cholesky, ~ u kappa per QR e SVD
    A, b, x_true = make_synthetic(300, 20, kappa=1e6)
    err_chol = rel(solve_normal_cholesky(A, b), x_true)
    err_qr = rel(solve_qr(A, b), x_true)
    err_svd = rel(solve_svd(A, b), x_true)
    assert err_qr < 1e-8 and err_svd < 1e-8
    assert err_chol > 100 * err_qr


def test_svd_norma_minima():
    # colonna duplicata: rango n - 1, la SVD troncata deve dare la soluzione di norma minima
    A, b, _ = make_synthetic(100, 8, kappa=10, residual=0.3)
    A = np.column_stack([A, A[:, 0]])
    x = solve_svd(A, b)
    assert rel(x, np.linalg.pinv(A) @ b) < 1e-12
    assert abs(x[0] - x[-1]) < 1e-12           # il peso si divide in parti uguali


# --- dati reali: rango deficiente (KWG^2, KWI^2) ------------------------------------

@pytest.fixture(scope="module")
def sgemm_campione():
    P, runs = load_sgemm()
    idx = np.random.default_rng(0).choice(len(P), 20_000, replace=False)
    return P[idx], target(runs[idx])


def test_rango_deficiente_sgemm(sgemm_campione):
    P, b = sgemm_campione
    A_full, _ = build_features(P, degree=2, drop_dependent=False)
    A, _ = build_features(P, degree=2)
    assert A_full.shape[1] == 116 and np.linalg.matrix_rank(A_full) == 114
    assert np.linalg.matrix_rank(A) == 114
    # la SVD troncata da' le stesse predizioni del modello senza le colonne dipendenti
    pred_full = A_full @ solve_svd(A_full, b)
    pred = A @ solve_svd(A, b)
    assert rel(pred_full, pred) < 1e-10
