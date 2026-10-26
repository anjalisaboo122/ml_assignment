import numpy as np
from common import load_data, stratified_split, accuracy, species_to_family_map


# ===================== Gaussian Naive Bayes =====================
class GaussianNB:
    """Gaussian Naive Bayes classifier.

    Parameters
    ----------
    var_smoothing : float
        Added to every per-class per-feature variance as
        epsilon = var_smoothing * max_j Var_train(x_j).
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        N, d = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)

        # epsilon = var_smoothing * max_j Var_train(x_j)
        eps = self.var_smoothing * np.var(X, axis=0, ddof=0).max()

        self.log_prior_ = np.empty(K)
        self.mean_ = np.empty((K, d))
        self.var_  = np.empty((K, d))

        for i, c in enumerate(self.classes_):
            mask = (y == c)
            Xc   = X[mask]
            nc   = mask.sum()
            self.log_prior_[i] = np.log(nc / N)
            self.mean_[i]      = Xc.mean(axis=0)
            self.var_[i]       = Xc.var(axis=0, ddof=0) + eps   # variance smoothing

        return self

    def predict(self, X):
        """Vectorised prediction — no Python loops over samples.

        log P(x|c) is computed for all (sample, class) pairs at once
        using numpy broadcasting over shape (n, K, d).
        """
        X = np.asarray(X, dtype=float)
        # diff: (n, K, d)
        diff = X[:, np.newaxis, :] - self.mean_[np.newaxis, :, :]
        # log-likelihood sum over features: (n, K)
        log_lik = -0.5 * np.sum(
            np.log(2.0 * np.pi * self.var_[np.newaxis, :, :])
            + diff ** 2 / self.var_[np.newaxis, :, :],
            axis=2
        )
        # log-posterior: (n, K)
        log_post = log_lik + self.log_prior_[np.newaxis, :]
        return self.classes_[np.argmax(log_post, axis=1)]


# ===================== sanity check =====================
def run_sanity_check():
    """One feature, class 0: {0,2}, class 1: {10,12}, equal priors.
    Means = 1 and 11, variances = 1 (+eps).
    x=3 -> class 0  (closer to mean 1),  x=9 -> class 1 (closer to mean 11).
    """
    X_toy = np.array([[0.0], [2.0], [10.0], [12.0]])
    y_toy = np.array([0, 0, 1, 1])
    nb = GaussianNB(var_smoothing=1e-9).fit(X_toy, y_toy)
    p3 = nb.predict([[3.0]])[0]
    p9 = nb.predict([[9.0]])[0]
    assert p3 == 0, f"Expected class 0 for x=3, got {p3}"
    assert p9 == 1, f"Expected class 1 for x=9, got {p9}"
    print("PASS sanity check: x=3 -> class 0,  x=9 -> class 1")


# ===================== main run =====================
def main():
    run_sanity_check()

    X, ys, yf, species_names, family_names = load_data()
    train, test = stratified_split(ys)
    Xtr, Xte = X[train], X[test]
    ytr, yte = ys[train], ys[test]
    fam_map = species_to_family_map(ytr, yf[train])

    nb = GaussianNB(var_smoothing=1e-9).fit(Xtr, ytr)

    tr_pred = nb.predict(Xtr)
    te_pred = nb.predict(Xte)
    tr_acc  = accuracy(ytr, tr_pred)
    te_acc  = accuracy(yte, te_pred)
    fam_acc = accuracy(yf[test], fam_map[te_pred])

    print(f"Gaussian NB  train={100*tr_acc:6.2f}%  test={100*te_acc:6.2f}%  "
          f"family(test)={100*fam_acc:6.2f}%")

    # machine-parsable RESULT line (no criterion for NB)
    print(f"RESULT|Gaussian Naive Bayes||{100*tr_acc:.2f}|{100*te_acc:.2f}|{100*fam_acc:.2f}")

    # --- Correlation analysis of the 22 training MFCCs ---
    corr = np.corrcoef(Xtr.T)   # (22, 22)
    print("\nCorrelation matrix of 22 training MFCCs (22x22):")
    np.set_printoptions(precision=3, suppress=True, linewidth=130)
    print(corr)

    upper_tri = corr[np.triu_indices(22, k=1)]
    print(f"\nMean |off-diagonal| correlation : {np.abs(upper_tri).mean():.3f}")
    print(f"Max  |off-diagonal| correlation : {np.abs(upper_tri).max():.3f}")
    print(
        "Note: MFCCs are derived from adjacent DCT coefficients of the same signal, "
        "so neighbouring coefficients tend to be correlated. Naive Bayes assumes "
        "conditional independence of features given the class, which is violated here. "
        "The model effectively double-counts shared information between correlated "
        "features, making its posterior probabilities overconfident. Despite this, "
        "the argmax decision (the predicted class) can still be competitive because "
        "the ranking of classes is preserved even when absolute probabilities are "
        "miscalibrated — a well-known property of Naive Bayes in practice."
    )


if __name__ == "__main__":
    main()
