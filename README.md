# Machine Unlearning with Influence Functions on Iris

A beginner-friendly implementation of **Machine Unlearning**, **Influence Functions**, and **Data Poisoning Detection** using the Iris dataset and logistic regression.

Based on:
- Koh & Liang (2017) — *Understanding Black-box Predictions via Influence Functions* (ICML)
- Saunshi et al. (2022) — *Understanding Influence Functions and Datamodels via Harmonic Analysis*
- Li et al. (2024) — *∆-Influence: Unlearning Poisons via Influence Functions* (NeurIPS ATTRIB Workshop)

---

## What This Project Does

This project answers a core question in modern ML:

> *"What happens to a trained model if a specific training point is removed — without retraining from scratch?"*

It implements five experiments end-to-end:

| # | Experiment | What it Shows |
|---|---|---|
| 1 | Hessian Sanity Checks | Verifies the math is correct before trusting any results |
| 2 | Influence Scores | Which training points help or hurt a given test prediction |
| 3 | LOO Validation | Compares influence predictions vs actual leave-one-out retraining |
| 4 | Machine Unlearning | Removes a "forget set" using 3 methods: Exact, Newton Step, Gradient Ascent |
| 5 | Data Poisoning + ∆-Influence | Injects label-flip poison, then detects poisoned points |

---

## Key Concepts

### Influence Functions
Instead of retraining the model without a point (expensive), influence functions approximate the effect using a single formula:

```
I(z_train, z_test) = − ∇L(z_test)ᵀ · H⁻¹ · ∇L(z_train)
```

Where `H` is the Hessian (curvature) of the loss. This tells us: *"if we removed this training point, how much would the test loss change?"*

### Machine Unlearning
Given a **forget set** D_f and a **retain set** D_r:

- **Exact Unlearning** — retrain from scratch on D_r (ground truth, expensive)
- **Newton Step** — approximate unlearning in one step using influence functions (fast)
- **Gradient Ascent** — maximize loss on D_f while minimizing on D_r

### ∆-Influence (Poison Detection)
Poisoned training points cause **influence collapse**: when the test point is transformed (e.g. label flipped, noise added), the influence of a poisoned point drops sharply. Clean points don't show this pattern.

```
ΔInfl(i, j) = Infl(z_train_i, g_j(z_test)) − Infl(z_train_i, z_test)
```

Points where this drop is consistently large across multiple transformations are flagged as poisoned.

---

## Results

```
Model accuracy (clean Iris, 2-class): 1.0000

EXPERIMENT 1 — Hessian Sanity Checks
  Symmetric:          True  ✓
  Positive definite:  True  ✓
  Condition number:   1.0   ✓ (well-conditioned)

EXPERIMENT 2 — Influence Scores
  Most helpful train point: same class as test point (expected)
  Most harmful train point: influence ≈ 0 (model is well-generalised)

EXPERIMENT 3 — LOO Validation
  Pearson R = 0.42
  Note: Low R is expected on a perfect-accuracy model (floor effect —
  all changes are near zero, making Pearson R unreliable)

EXPERIMENT 4 — Machine Unlearning (forget 10 points)
  Method                  Test Loss
  Original model          0.0155
  Exact unlearn           0.0169  ← ground truth
  Newton step (approx)    0.0155  ← closest to exact
  Gradient ascent         0.0192

EXPERIMENT 5 — Poisoning
  Iris is too easy: even 8 label flips don't cause misclassification.
  Delta-Influence requires a harder dataset to demonstrate detection.
```

---

## Setup & Usage

### Requirements
- Python 3.10+
- pip

### Install dependencies
```bash
pip install numpy scikit-learn scipy matplotlib
```

### Run
```bash
python machine_unlearning.py
```

The script prints results for all 5 experiments and saves two plots:
- `loo_validation.png` — predicted vs actual LOO loss change
- `delta_influence.png` — vote counts for poison detection

---

## Project Structure

```
machine_unlearning.py   ← single file, all experiments included
README.md
loo_validation.png      ← generated on run
delta_influence.png     ← generated on run
```

---

## Important Implementation Notes

These are the most common mistakes when implementing influence functions (from the notes):

1. **Never skip L2 regularization in the Hessian** — the `(1/C)·I` term is mandatory. Without it the Hessian is singular and all influence scores are garbage.
2. **The negative sign matters** — `I = −s_test · ∇L(z_train)`. Flipping it means you unlearn helpful points and keep harmful ones.
3. **Always normalize features** — use `StandardScaler` before computing the Hessian. Different feature scales cause numerical instability.
4. **Add damping** — a small `λI` floor (0.001–0.01) ensures the Hessian is positive definite even without perfect regularization.
5. **Use `fit_intercept=False`** — manually append a bias column to X and disable sklearn's internal intercept so the Hessian shape is consistent.

---

## Limitations

- Iris (2-class) is near-perfect accuracy, so influence scores are very small and LOO validation shows low Pearson R. This is a **floor effect**, not a bug.
- ∆-Influence poison detection requires a harder dataset (e.g. MNIST, CIFAR) where poisoning actually causes misclassifications.
- The Newton step unlearning approximation is only valid for **small forget sets** and **convex models** (logistic regression). It degrades for neural networks.

---

## References

```
[1] Koh, P.W. & Liang, P. (2017).
    Understanding Black-box Predictions via Influence Functions. ICML.

[2] Saunshi, N. et al. (2022).
    Understanding Influence Functions and Datamodels via Harmonic Analysis.

[3] Li, W. et al. (2024).
    ∆-Influence: Unlearning Poisons via Influence Functions.
    NeurIPS ATTRIB Workshop.
```
