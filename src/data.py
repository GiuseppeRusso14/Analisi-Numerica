"""Caricamento del dataset SGEMM e costruzione della matrice dei dati A.

Problema: min_x f(x) = 1/2 ||Ax - b||^2, con b = log(tempo medio di esecuzione).
Ogni riga di A descrive una configurazione del kernel: colonna di 1 (intercetta),
parametri e, nei modelli più ricchi, i loro prodotti (interazioni).

Modelli:
    M1 (degree=1): intercetta + 14 parametri                      -> 15 colonne
    M2 (degree=2): M1 + 91 prodotti a coppie + quadrati            -> 114 colonne
    M3 (degree=3): M2 + 364 prodotti di tre parametri distinti     -> 478 colonne
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sgemm_product.csv"

# 10 parametri ordinali (potenze di 2) e 4 binari {0, 1}
ORDINAL = ["MWG", "NWG", "KWG", "MDIMC", "NDIMC", "MDIMA", "NDIMB", "KWI", "VWM", "VWN"]
BINARY = ["STRM", "STRN", "SA", "SB"]
PARAMS = ORDINAL + BINARY
RUNS = ["Run1 (ms)", "Run2 (ms)", "Run3 (ms)", "Run4 (ms)"]


def load_sgemm(path=DATA_PATH):
    """Legge il CSV e restituisce (P, runs).

    P    : (m, 14) parametri delle configurazioni, nell'ordine di PARAMS
    runs : (m, 4)  le quattro misure del tempo, in millisecondi
    """
    df = pd.read_csv(path)
    P = df[PARAMS].to_numpy(dtype=float)
    runs = df[RUNS].to_numpy(dtype=float)
    return P, runs


def target(runs):
    """b = log(media delle 4 misure)."""
    return np.log(runs.mean(axis=1))


def build_features(P, degree=2, drop_dependent=True):
    """Costruisce la matrice A del modello M1, M2 o M3 e i nomi delle colonne.

    Quadrati: solo dei parametri ordinali (per un binario t^2 = t, colonna già presente).
    Un parametro con due soli valori {a, b} soddisfa t^2 = (a + b) t - ab, quindi il suo
    quadrato è combinazione lineare di t e della colonna di 1: è il caso di KWG e KWI.
    Con drop_dependent=True questi quadrati vengono esclusi (A di rango pieno);
    con drop_dependent=False restano, e A ha rango deficiente (esperimento E1).
    """
    m, p = P.shape
    cols = [np.ones(m)]
    names = ["1"]

    cols += [P[:, j] for j in range(p)]
    names += PARAMS

    if degree >= 2:
        for i, j in combinations(range(p), 2):
            cols.append(P[:, i] * P[:, j])
            names.append(f"{PARAMS[i]}*{PARAMS[j]}")
        for j, name in enumerate(ORDINAL):
            if drop_dependent and len(np.unique(P[:, j])) <= 2:
                continue
            cols.append(P[:, j] ** 2)
            names.append(f"{name}^2")

    if degree >= 3:
        for i, j, k in combinations(range(p), 3):
            cols.append(P[:, i] * P[:, j] * P[:, k])
            names.append(f"{PARAMS[i]}*{PARAMS[j]}*{PARAMS[k]}")

    return np.column_stack(cols), names


def standardize(A, mu=None, sigma=None):
    """Standardizzazione z-score delle colonne di A, intercetta esclusa.

    Se mu e sigma non sono dati si calcolano da A (dati di training); per i dati di
    test si passano quelli del training, così il test non influenza il modello.
    Restituisce (A_standardizzata, mu, sigma).
    """
    if mu is None:
        mu = A.mean(axis=0)
        sigma = A.std(axis=0)
        mu[0], sigma[0] = 0.0, 1.0  # la colonna di 1 resta com'è
    return (A - mu) / sigma, mu, sigma


def make_synthetic(m, n, kappa, residual=0.0, seed=0):
    """Problema sintetico con condizionamento e soluzione noti (solo per i test).

    A = U diag(s) V^T con U, V a colonne ortonormali casuali e valori singolari
    s logaritmicamente equispaziati tra 1 e 1/kappa, quindi kappa_2(A) = kappa.
    b = A x_true + r, con r ortogonale a range(A) e ||r|| = residual * ||A x_true||:
    x_true è la soluzione esatta dei minimi quadrati, qualunque sia residual.
    Restituisce (A, b, x_true).
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.logspace(0, -np.log10(kappa), n)
    A = U @ np.diag(s) @ V.T
    x_true = rng.standard_normal(n)
    b = A @ x_true
    if residual > 0:
        r = rng.standard_normal(m)
        r -= U @ (U.T @ r)  # proiezione sul complemento ortogonale di range(A)
        b += residual * np.linalg.norm(b) / np.linalg.norm(r) * r
    return A, b, x_true


def train_test_split(m, train_frac=0.8, seed=0):
    """Indici di training e di test, scelti a caso con seed fisso."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    n_train = int(round(train_frac * m))
    return perm[:n_train], perm[n_train:]


if __name__ == "__main__":
    # Riepilogo: dimensioni, rango, condizionamento e R^2 di riferimento (np.linalg.lstsq)
    P, runs = load_sgemm()
    b = target(runs)
    t = runs.mean(axis=1)
    noise = (runs.std(axis=1) / t).mean()
    print(f"configurazioni: {P.shape[0]}   parametri: {P.shape[1]}")
    print(f"tempo min / mediana / max: {t.min():.2f} / {np.median(t):.2f} / {t.max():.2f} ms")
    print(f"rumore tra le 4 misure (std relativa media): {100 * noise:.2f}%")

    itr, ite = train_test_split(len(b))
    for degree in (1, 2):
        A, names = build_features(P, degree)
        As, mu, sigma = standardize(A[itr])
        At, _, _ = standardize(A[ite], mu, sigma)
        x = np.linalg.lstsq(As, b[itr], rcond=None)[0]
        r2 = 1 - np.sum((b[ite] - At @ x) ** 2) / np.sum((b[ite] - b[ite].mean()) ** 2)
        print(f"\nM{degree}: A = {A.shape[0]} x {A.shape[1]}")
        print(f"  kappa(A) grezzo: {np.linalg.cond(A[itr]):.2e}   standardizzato: {np.linalg.cond(As):.1f}")
        print(f"  R^2 sul test: {r2:.3f}")

    A_full, _ = build_features(P, 2, drop_dependent=False)
    print(f"\nM2 con KWG^2 e KWI^2: {A_full.shape[1]} colonne, rango {np.linalg.matrix_rank(A_full)}")
