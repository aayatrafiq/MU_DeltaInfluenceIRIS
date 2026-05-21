"""
Machine Unlearning — Complete Implementation
Based on your notes: Influence Functions, Data Poisoning, Unlearning on Iris

HOW TO RUN:
  1. Install dependencies (run this once in your terminal):
       pip install numpy scikit-learn scipy matplotlib

  2. Run this file:
       python machine_unlearning.py

The script runs 5 experiments in order and prints results for each.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — HELPER MATH FUNCTIONS
# These are the building blocks used everywhere else.
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(t):
    """
    Squashes any number into the range (0, 1).
    Used to turn a raw model score into a probability.
    np.clip prevents overflow for very large/small values.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(t, -500, 500)))


def compute_gradient(x, y, theta):
    """
    Gradient of the logistic loss for ONE training point.

    Args:
        x     : feature vector for one point, shape (p,)
        y     : label, 0 or 1
        theta : model weights, shape (p,)

    Returns:
        gradient vector, shape (p,)

    Formula: ∇L = (sigmoid(θ·x) − y) · x
    Intuition: if the model is wrong (high error), the gradient is large.
    """
    p = sigmoid(np.dot(theta, x))   # predicted probability
    error = p - y                   # how wrong we are
    return error * x                # scale features by the error


def compute_hessian(X, y, theta, C=1.0, damping=0.01):
    """
    The Hessian matrix — measures the curvature of the loss surface.
    Shape: (p, p) where p = number of features.

    Args:
        X       : training features, shape (n, p)
        y       : training labels, shape (n,)
        theta   : model weights, shape (p,)
        C       : regularization parameter from sklearn (default 1.0)
        damping : small value added for numerical stability

    Formula: H = (1/n) XᵀWX + (1/C)·I + damping·I
    where W = diag(p_i · (1 - p_i)), the per-point uncertainty weights.

    CRITICAL: The (1/C)·I term is the L2 regularization.
    If you forget it, the Hessian is wrong and everything breaks.
    """
    n, p = X.shape
    probs = sigmoid(X @ theta)          # predicted probabilities, shape (n,)
    W = probs * (1 - probs)             # uncertainty weights, shape (n,)

    # Weighted sum of outer products: (1/n) Xᵀ diag(W) X
    H = (X.T * W) @ X / n              # shape (p, p)

    H += (1.0 / C) * np.eye(p)         # L2 regularization — DO NOT SKIP
    H += damping * np.eye(p)           # numerical stability floor
    return H


def compute_influence_score(x_train, y_train, x_test, y_test, theta, H_inv):
    """
    Influence of ONE training point on ONE test point's loss.

    Formula: I = −s_test · ∇L(z_train, θ)
    where s_test = H⁻¹ · ∇L(z_test, θ)

    Interpretation:
      Large negative → removing this train point HURTS test loss (helpful point)
      Large positive → removing this train point HELPS test loss (harmful point)
      Near zero      → this train point barely affects this test point
    """
    g_test  = compute_gradient(x_test,  y_test,  theta)  # shape (p,)
    g_train = compute_gradient(x_train, y_train, theta)  # shape (p,)
    s_test  = H_inv @ g_test                              # shape (p,)
    return -float(s_test @ g_train)                       # scalar


def compute_test_loss(model, X_test, y_test):
    """Average log-loss (cross-entropy) on the test set."""
    probs = model.predict_proba(X_test)[:, 1]            # P(y=1|x)
    probs = np.clip(probs, 1e-10, 1 - 1e-10)             # avoid log(0)
    losses = -(y_test * np.log(probs) + (1 - y_test) * np.log(1 - probs))
    return float(np.mean(losses))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATA LOADING
# Load Iris, keep only 2 classes (binary classification), normalize features.
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Loads the Iris dataset and prepares it for binary logistic regression.

    Steps:
      1. Load all 150 Iris samples (3 classes, 4 features each)
      2. Keep only classes 0 and 1 (100 samples total)
      3. Split into train (80%) and test (20%)
      4. StandardScale features — makes all features have mean=0, std=1
         (IMPORTANT: prevents ill-conditioned Hessian)
      5. Append a bias column (column of 1s) so we don't need fit_intercept
    """
    X, y = load_iris(return_X_y=True)
    mask = y != 2                                  # drop class 2
    X, y = X[mask], y[mask]                        # 100 samples remain

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)        # fit on train only
    X_test  = scaler.transform(X_test)             # apply same scale to test

    # Append bias column: [x1, x2, x3, x4] → [x1, x2, x3, x4, 1]
    X_train = np.hstack([X_train, np.ones((len(X_train), 1))])
    X_test  = np.hstack([X_test,  np.ones((len(X_test),  1))])

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, C=1.0):
    """
    Train logistic regression using sklearn.
    fit_intercept=False because we manually added a bias column above.
    max_iter=5000 ensures convergence.
    """
    model = LogisticRegression(C=C, max_iter=5000, fit_intercept=False)
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — EXPERIMENT 1: Sanity Checks on the Hessian
# Before trusting influence scores, verify the Hessian is mathematically valid.
# ─────────────────────────────────────────────────────────────────────────────

def experiment_sanity_checks(X_train, y_train, model):
    print("\n" + "="*60)
    print("EXPERIMENT 1: Hessian Sanity Checks")
    print("="*60)

    theta = model.coef_[0]
    H = compute_hessian(X_train, y_train, theta, C=1.0)

    # Check 1: Symmetry (H should equal its transpose)
    is_symmetric = np.allclose(H, H.T)
    print(f"\n[1] Hessian is symmetric: {is_symmetric}  (should be True)")

    # Check 2: Positive definite (all eigenvalues > 0)
    eigenvalues = np.linalg.eigvals(H)
    is_pd = bool(np.all(eigenvalues > 0))
    print(f"[2] Hessian is positive definite: {is_pd}  (should be True)")

    # Check 3: Condition number (ratio of largest to smallest eigenvalue)
    cond = np.linalg.cond(H)
    print(f"[3] Condition number: {cond:.1f}  (should be < 1,000,000)")
    if cond > 1e6:
        print("    WARNING: High condition number! Try adding more damping.")
    else:
        print("    Good — Hessian is well-conditioned.")

    # Check 4: Gradient near zero at optimum (model has converged)
    n = len(X_train)
    grad_sum = sum(compute_gradient(X_train[i], y_train[i], theta)
                   for i in range(n))
    grad_norm = np.linalg.norm(grad_sum / n)
    print(f"[4] Gradient norm at optimum: {grad_norm:.6f}  (should be < 0.01)")

    print("\nAll checks passed!" if (is_symmetric and is_pd and cond < 1e6 and grad_norm < 0.1)
          else "\nSome checks failed — review warnings above.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — EXPERIMENT 2: Influence Scores
# Compute which training points most influence each test point's loss.
# ─────────────────────────────────────────────────────────────────────────────

def experiment_influence_scores(X_train, y_train, X_test, y_test, model):
    print("\n" + "="*60)
    print("EXPERIMENT 2: Influence Scores")
    print("="*60)

    theta = model.coef_[0]
    H     = compute_hessian(X_train, y_train, theta, C=1.0)
    H_inv = np.linalg.inv(H)    # safe for small p (p=5 here)

    # Pick test point 0 as our target
    x_te, y_te = X_test[0], y_test[0]
    print(f"\nTarget test point — label: {y_te}")

    # Compute influence of every training point on this test point
    scores = np.array([
        compute_influence_score(X_train[i], y_train[i], x_te, y_te, theta, H_inv)
        for i in range(len(X_train))
    ])

    # Top 3 most helpful (large negative score)
    helpful_idx = np.argsort(scores)[:3]
    print("\nTop 3 HELPFUL training points (removing them hurts test loss):")
    for rank, i in enumerate(helpful_idx, 1):
        print(f"  #{rank}: train[{i}]  label={y_train[i]}  influence={scores[i]:.4f}")

    # Top 3 most harmful (large positive score)
    harmful_idx = np.argsort(scores)[-3:][::-1]
    print("\nTop 3 HARMFUL training points (removing them helps test loss):")
    for rank, i in enumerate(harmful_idx, 1):
        print(f"  #{rank}: train[{i}]  label={y_train[i]}  influence={scores[i]:.4f}")

    return scores, H_inv, theta


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EXPERIMENT 3: LOO Validation (the most important test)
# Compare influence predictions against actual leave-one-out retraining.
# A Pearson R > 0.9 means your Hessian is correct.
# ─────────────────────────────────────────────────────────────────────────────

def experiment_loo_validation(X_train, y_train, X_test, y_test,
                               model, influence_scores, H_inv, theta, k=20):
    print("\n" + "="*60)
    print("EXPERIMENT 3: Leave-One-Out (LOO) Validation")
    print("="*60)
    print(f"Comparing influence predictions vs actual retraining for top {k} points...")

    x_te, y_te   = X_test[0], y_test[0]
    full_loss    = compute_test_loss(model, X_test, y_test)
    top_k_idx    = np.argsort(np.abs(influence_scores))[-k:]

    predicted, actual = [], []

    for i in top_k_idx:
        # --- Influence prediction ---
        # Removing point i ≈ upweighting by ε = -1/n
        pred_change = -influence_scores[i] / len(X_train)
        predicted.append(pred_change)

        # --- Actual LOO retraining ---
        mask      = np.arange(len(X_train)) != i
        loo_model = train_model(X_train[mask], y_train[mask])
        loo_loss  = compute_test_loss(loo_model, X_test, y_test)
        actual.append(loo_loss - full_loss)

    predicted = np.array(predicted)
    actual    = np.array(actual)

    r, pval = pearsonr(predicted, actual)
    print(f"\nPearson R = {r:.4f}  (target: > 0.90)")
    if r > 0.90:
        print("PASS — Influence function is accurate!")
    elif r > 0.70:
        print("PARTIAL — Reasonable correlation, but check your Hessian.")
    else:
        print("FAIL — Low correlation. Check: regularization, normalization, damping.")

    # Plot predicted vs actual
    plt.figure(figsize=(6, 5))
    plt.scatter(predicted, actual, color='steelblue', alpha=0.7, edgecolors='white')
    # Diagonal reference line
    lims = [min(predicted.min(), actual.min()), max(predicted.max(), actual.max())]
    plt.plot(lims, lims, 'r--', label='Perfect prediction')
    plt.xlabel("Influence-predicted Δloss")
    plt.ylabel("Actual Δloss (LOO retraining)")
    plt.title(f"LOO Validation  (Pearson R = {r:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loo_validation.png", dpi=120)
    plt.show()
    print("Plot saved as: loo_validation.png")

    return r


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — EXPERIMENT 4: Machine Unlearning
# Remove a "forget set" from the model without full retraining.
# Compare three methods: Exact (retrain), Newton Step, Gradient Ascent.
# ─────────────────────────────────────────────────────────────────────────────

def approximate_unlearn_newton(theta, X_forget, y_forget, H_inv, n_total):
    """
    Approximate unlearning via a single Newton step.

    Formula: θ_new = θ + (1/n) · H⁻¹ · Σ_{D_f} ∇L(z, θ)

    Intuition: we add back the gradient contribution of the forget points,
    which is equivalent to "undoing" what those points did to the weights.
    """
    total_grad = sum(
        compute_gradient(X_forget[i], y_forget[i], theta)
        for i in range(len(X_forget))
    )
    delta_theta = (1.0 / n_total) * H_inv @ total_grad
    return theta + delta_theta


def experiment_unlearning(X_train, y_train, X_test, y_test, model, H_inv, theta):
    print("\n" + "="*60)
    print("EXPERIMENT 4: Machine Unlearning")
    print("="*60)

    # --- Define forget set: 10 random training points ---
    rng        = np.random.default_rng(seed=0)
    forget_idx = rng.choice(len(X_train), size=10, replace=False)
    retain_idx = np.array([i for i in range(len(X_train)) if i not in forget_idx])

    X_forget, y_forget = X_train[forget_idx], y_train[forget_idx]
    X_retain, y_retain = X_train[retain_idx], y_train[retain_idx]

    print(f"\nForget set size: {len(forget_idx)}")
    print(f"Retain set size: {len(retain_idx)}")

    # ── Method 1: Exact Unlearning (ground truth) ─────────────────────────
    exact_model = train_model(X_retain, y_retain)
    exact_acc   = exact_model.score(X_test, y_test)
    exact_loss  = compute_test_loss(exact_model, X_test, y_test)

    # ── Method 2: Newton Step (influence function approximation) ──────────
    theta_newton = approximate_unlearn_newton(theta, X_forget, y_forget,
                                              H_inv, len(X_train))
    # Wrap theta_newton in a temporary model-like object for evaluation
    newton_model          = train_model(X_retain, y_retain)  # get structure
    newton_model.coef_[0] = theta_newton                     # replace weights
    newton_acc            = newton_model.score(X_test, y_test)
    newton_loss           = compute_test_loss(newton_model, X_test, y_test)

    # ── Method 3: Gradient Ascent on forget set (simple fine-tune) ────────
    theta_ga = theta.copy()
    lr       = 0.01
    for step in range(50):   # 50 gradient ascent steps
        for i in range(len(X_forget)):
            # Ascent on forget: MAXIMIZE loss on forget points
            theta_ga += lr * compute_gradient(X_forget[i], y_forget[i], theta_ga)
        for i in range(len(X_retain)):
            # Descent on retain: MINIMIZE loss on retain points
            theta_ga -= (lr / len(X_retain)) * compute_gradient(
                X_retain[i], y_retain[i], theta_ga)

    ga_model          = train_model(X_retain, y_retain)
    ga_model.coef_[0] = theta_ga
    ga_acc            = ga_model.score(X_test, y_test)
    ga_loss           = compute_test_loss(ga_model, X_test, y_test)

    # ── Original model baseline ────────────────────────────────────────────
    orig_acc  = model.score(X_test, y_test)
    orig_loss = compute_test_loss(model, X_test, y_test)

    # ── Print results table ────────────────────────────────────────────────
    print(f"\n{'Method':<22} {'Test Acc':>10} {'Test Loss':>12}")
    print("-" * 46)
    print(f"{'Original model':<22} {orig_acc:>10.4f} {orig_loss:>12.4f}")
    print(f"{'Exact unlearn (retrain)':<22} {exact_acc:>10.4f} {exact_loss:>12.4f}")
    print(f"{'Newton step (approx)':<22} {newton_acc:>10.4f} {newton_loss:>12.4f}")
    print(f"{'Gradient ascent':<22} {ga_acc:>10.4f} {ga_loss:>12.4f}")

    print("\nWhat to look for:")
    print("  • Exact unlearn = ground truth. Newton step should be close to it.")
    print("  • All methods should keep reasonable test accuracy (not drop to 50%).")

    return forget_idx, retain_idx


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — EXPERIMENT 5: Data Poisoning + Delta-Influence Detection
# Inject label-flip poison points, then detect them using ∆-Influence.
# ─────────────────────────────────────────────────────────────────────────────

def flip_label(x, y):
    """Transformation 1: flip the test label (y=0→1, y=1→0)."""
    return x, 1 - y

def add_noise(x, y):
    """Transformation 2: add tiny Gaussian noise to features."""
    return x + 0.05 * np.random.randn(*x.shape), y

def scale_features(x, y):
    """Transformation 3: randomly scale each feature by 0.9–1.1×."""
    scale = np.random.uniform(0.9, 1.1, size=x.shape)
    return x * scale, y


def experiment_poisoning(X_train, y_train, X_test, y_test):
    print("\n" + "="*60)
    print("EXPERIMENT 5: Data Poisoning + Delta-Influence Detection")
    print("="*60)

    # ── Step 1: Inject label-flip poison ──────────────────────────────────
    n_poison   = 8
    rng        = np.random.default_rng(seed=7)
    poison_idx = rng.choice(len(X_train), size=n_poison, replace=False)

    y_poisoned            = y_train.copy()
    y_poisoned[poison_idx] = 1 - y_train[poison_idx]   # flip labels

    print(f"\nInjected {n_poison} poison points at indices: {sorted(poison_idx)}")

    # ── Step 2: Train on poisoned data ────────────────────────────────────
    model_p = train_model(X_train, y_poisoned)
    theta_p = model_p.coef_[0]

    clean_acc   = train_model(X_train, y_train).score(X_test, y_test)
    poisoned_acc = model_p.score(X_test, y_test)
    print(f"Accuracy before poisoning: {clean_acc:.4f}")
    print(f"Accuracy after  poisoning: {poisoned_acc:.4f}")

    # ── Step 3: Find a test point the poisoned model gets wrong ───────────
    preds   = model_p.predict(X_test)
    bad_idx = np.where(preds != y_test)[0]

    if len(bad_idx) == 0:
        print("\nNo misclassified test points found. Try more poison points.")
        return

    x_bad, y_bad = X_test[bad_idx[0]], y_test[bad_idx[0]]
    print(f"\nTarget misclassified test point index: {bad_idx[0]}  (true label: {y_bad})")

    # ── Step 4: Compute Delta-Influence ───────────────────────────────────
    H_p     = compute_hessian(X_train, y_poisoned, theta_p, C=1.0)
    H_p_inv = np.linalg.inv(H_p)

    transformations = [flip_label, add_noise, scale_features]

    # Base influence scores
    base_scores = np.array([
        compute_influence_score(X_train[i], y_poisoned[i],
                                x_bad, y_bad, theta_p, H_p_inv)
        for i in range(len(X_train))
    ])

    # For each transformation, count how many give a large negative drop
    vote_counts = np.zeros(len(X_train))
    tau = -0.01   # threshold: a drop below this counts as "collapsed"

    for transform_fn in transformations:
        x_aug, y_aug = transform_fn(x_bad, y_bad)
        aug_scores = np.array([
            compute_influence_score(X_train[i], y_poisoned[i],
                                    x_aug, y_aug, theta_p, H_p_inv)
            for i in range(len(X_train))
        ])
        delta = aug_scores - base_scores
        vote_counts += (delta < tau).astype(int)

    # Flag points where MOST transformations caused influence collapse
    n_tol    = 1    # allow 1 transformation to not trigger
    flagged  = vote_counts >= (len(transformations) - n_tol)
    detected = np.where(flagged)[0]

    # ── Step 5: Evaluate detection ─────────────────────────────────────────
    poison_set   = set(poison_idx.tolist())
    detected_set = set(detected.tolist())

    true_positives  = poison_set & detected_set
    false_positives = detected_set - poison_set
    false_negatives = poison_set - detected_set

    precision = len(true_positives) / max(len(detected_set), 1)
    recall    = len(true_positives) / max(len(poison_set), 1)

    print(f"\nActual poison indices:   {sorted(poison_set)}")
    print(f"Detected indices:        {sorted(detected_set)}")
    print(f"\nTrue  positives (correct detections): {sorted(true_positives)}")
    print(f"False positives (false alarms):        {sorted(false_positives)}")
    print(f"False negatives (missed poisons):      {sorted(false_negatives)}")
    print(f"\nPrecision: {precision:.2f}  (fraction of detections that are real poisons)")
    print(f"Recall:    {recall:.2f}  (fraction of real poisons that were caught)")

    # Plot vote counts
    colors = ['red' if i in poison_set else 'steelblue' for i in range(len(X_train))]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(X_train)), vote_counts, color=colors, alpha=0.8)
    plt.axhline(len(transformations) - n_tol - 0.5, color='orange',
                linestyle='--', label='Detection threshold')
    plt.xlabel("Training point index")
    plt.ylabel("Number of transformations causing influence collapse")
    plt.title("Delta-Influence: Vote Counts (red = actual poison)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("delta_influence.png", dpi=120)
    plt.show()
    print("Plot saved as: delta_influence.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run all experiments in order
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading and preparing Iris dataset...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train size: {len(X_train)},  Test size: {len(X_test)},  Features: {X_train.shape[1]}")

    print("\nTraining logistic regression model...")
    model = train_model(X_train, y_train)
    theta = model.coef_[0]
    print(f"Model accuracy: {model.score(X_test, y_test):.4f}")

    # Run all 5 experiments
    experiment_sanity_checks(X_train, y_train, model)

    scores, H_inv, theta = experiment_influence_scores(
        X_train, y_train, X_test, y_test, model)

    experiment_loo_validation(
        X_train, y_train, X_test, y_test, model, scores, H_inv, theta, k=20)

    experiment_unlearning(
        X_train, y_train, X_test, y_test, model, H_inv, theta)

    experiment_poisoning(X_train, y_train, X_test, y_test)

    print("\n" + "="*60)
    print("All experiments complete!")
    print("="*60)


if __name__ == "__main__":
    main()