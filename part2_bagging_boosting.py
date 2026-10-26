import time
import numpy as np
from common import load_data, stratified_split, accuracy, species_to_family_map
from part1_decision_tree import DecisionTree
 
 
def majority_vote(preds, n_classes):
    #preds: (T, n) array of class labels. Returns (n,) winners.
    #Ties go to the smallest label because argmax returns the first max.
    T, n = preds.shape
    counts = np.zeros((n, n_classes), dtype=int)
    for t in range(T):
        counts[np.arange(n), preds[t]] += 1
    return counts.argmax(axis=1)
 
 
# 2.1 Bagging 
class Bagging:
    def __init__(self, n_estimators=25, criterion="gini", seed=42):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.seed = seed
 
    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        N = len(y)
        self.n_classes_ = int(y.max()) + 1
        rng = np.random.RandomState(self.seed)          # ONE rng for the whole ensemble
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.randint(0, N, size=N)              # bootstrap: N rows with replacement
            tree = DecisionTree(self.criterion, max_depth=None, min_samples_split=2)
            self.trees_.append(tree.fit(X[idx], y[idx]))
        return self
 
    def predict(self, X):
        preds = np.array([t.predict(X) for t in self.trees_])
        return majority_vote(preds, self.n_classes_)
 
 
# 2.2 AdaBoost (multi-class SAMME) 
class AdaBoostSAMME:
    def __init__(self, n_estimators=50, max_depth=3, criterion="gini"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
 
    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        N = len(y)
        K = int(y.max()) + 1
        self.n_classes_ = K
        w = np.full(N, 1.0 / N)                           # step 1: equal weights
        self.trees_, self.alphas_ = [], []
        for _ in range(self.n_estimators):
            tree = DecisionTree(self.criterion, max_depth=self.max_depth)
            tree.fit(X, y, sample_weight=w)               # step 2: weighted fit, no resampling
            wrong = tree.predict(X) != y
            eps = np.clip(w[wrong].sum(), 1e-10, 1 - 1e-10)   # step 3: weighted error
            if eps >= 1 - 1.0 / K:                        # no better than random: discard, stop
                break
            alpha = np.log((1 - eps) / eps) + np.log(K - 1)    # step 4: amount of say
            w = w * np.exp(alpha * wrong)                 # step 5: only wrong ones grow
            w = w / w.sum()
            self.trees_.append(tree)
            self.alphas_.append(alpha)
        return self
 
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.zeros((len(X), self.n_classes_))
        rows = np.arange(len(X))
        for tree, alpha in zip(self.trees_, self.alphas_):
            scores[rows, tree.predict(X)] += alpha        # each tree adds alpha to its vote
        return scores.argmax(axis=1)                      # step 6: H(x) = argmax_k sum alpha
 
 
# main run 
def report(name, crit, model, Xtr, ytr, Xte, yte, yf_te, fam_map, secs, in_matrix=True):
    tr = accuracy(ytr, model.predict(Xtr))
    te_pred = model.predict(Xte)
    te = accuracy(yte, te_pred)
    fam = accuracy(yf_te, fam_map[te_pred])
    print(f"{name:16s} {crit:8s} fit={secs:5.1f}s  train={100*tr:6.2f}%  "
          f"test={100*te:6.2f}%  family(test)={100*fam:6.2f}%")
    if in_matrix:
        print(f"RESULT|{name}|{crit}|{100*tr:.2f}|{100*te:.2f}|{100*fam:.2f}")
    return te
 
 
def main():
    X, ys, yf, _, _ = load_data()
    train, test = stratified_split(ys)
    Xtr, Xte, ytr, yte = X[train], X[test], ys[train], ys[test]
    fam_map = species_to_family_map(ytr, yf[train])
    args = (Xtr, ytr, Xte, yte, yf[test], fam_map)
 
    for crit in ("gini", "entropy"):
        print(f"\n=== criterion = {crit} ===")
        t0 = time.perf_counter(); m = Bagging(25, crit).fit(Xtr, ytr)
        report("Bagging", crit, m, *args, time.perf_counter() - t0)
 
        t0 = time.perf_counter(); m = AdaBoostSAMME(50, 3, crit).fit(Xtr, ytr)
        report("AdaBoost", crit, m, *args, time.perf_counter() - t0)
        print(f"  SAMME rounds kept: {len(m.trees_)}")
 
        t0 = time.perf_counter(); m = AdaBoostSAMME(50, 1, crit).fit(Xtr, ytr)
        report("AdaBoost stumps", crit, m, *args, time.perf_counter() - t0, in_matrix=False)
        print(f"  stump rounds kept: {len(m.trees_)}  (not in matrix)")
 
 
 
if __name__ == "__main__":
    main()