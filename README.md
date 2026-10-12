# Auto-tuning di kernel GPU con modelli ai minimi quadrati: metodi diretti vs metodi iterativi

Progetto d'esame per il corso **Numerical Methods for Scientific Computing**
(LM-18 Informatica, Università di Catania, prof. S. Boscarino), A.A. 2026/2027.

Autore: Giuseppe Russo

## Il problema

I kernel delle librerie GPU (ad esempio il prodotto di matrici SGEMM) hanno molti parametri di
configurazione: tiling, dimensione del workgroup, larghezza dei vettori, uso della memoria locale.
La configurazione migliore dipende dall'hardware e può essere molte volte più veloce di una
qualsiasi, ma provarle tutte (**auto-tuning esaustivo**) richiede moltissimi benchmark.

Un **modello di prestazioni** stima il tempo di esecuzione a partire dai parametri. Si addestra su
poche configurazioni misurate, si predice il tempo di tutte le altre e si eseguono solo le più
promettenti.

Qui il modello è lineare nei parametri e nelle loro interazioni, e si ottiene risolvendo un
problema ai minimi quadrati

$$\min_{x} f(x) = \tfrac12 \|Ax - b\|_2^2, \qquad b = \log(\text{tempo di esecuzione}).$$

## Metodi confrontati

- **Metodi diretti (forma chiusa):** equazioni normali con Cholesky, QR di Householder, SVD.
- **Metodi iterativi:** Richardson (gradiente a passo fisso, con passo scelto tramite metodo delle
  potenze e Gershgorin), steepest descent con passo ottimo, gradiente coniugato, quasi-Newton BFGS,
  ADAM.

Il collegamento tra le due famiglie è la condizione di ottimalità
$\nabla f(x) = A^T(Ax-b) = 0$, cioè le equazioni normali $A^TA\,x = A^Tb$.

I metodi si confrontano per accuratezza (al variare del condizionamento), velocità di convergenza,
costo computazionale e qualità del tuning ottenuto.

## Struttura

```
data/          dataset SGEMM GPU kernel performance (UCI)
src/           implementazione dei metodi
experiments/   esperimenti E1–E4
tests/         test automatici (pytest)
figures/       grafici generati
results/       tabelle (CSV) generate
relazione/     relazione tecnica (LaTeX + PDF)
presentazione/ slide (LaTeX Beamer + PDF)
```

## Installazione

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Esecuzione

```bash
pytest tests/          # verifica dei metodi
python run_all.py      # esegue tutti gli esperimenti, salva figure/ e results/
```

## Dataset

*SGEMM GPU kernel performance*, UCI Machine Learning Repository, licenza CC BY 4.0
(https://doi.org/10.24432/C5MK70). Contiene i tempi di esecuzione (4 misure) del prodotto di
matrici 2048×2048 con il kernel `gemm_fast` della libreria CLTune su GPU NVIDIA GTX 680, per
241.600 configurazioni valide di 14 parametri. La descrizione originale è in
`data/README_dataset.txt`.

- R. Ballester-Ripoll, E. G. Paredes, R. Pajarola, *Sobol Tensor Trains for Global Sensitivity
  Analysis*, Reliability Engineering & System Safety, 2019.
- C. Nugteren, V. Codreanu, *CLTune: A Generic Auto-Tuner for OpenCL Kernels*, MCSoC 2015, IEEE.
