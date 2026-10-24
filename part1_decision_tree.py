import time
import numpy as np
from common import load_data, stratified_split, accuracy, species_to_family_map

TOL = 1e-12   # tiny number: treats float differences smaller than this as "equal"


# ===================== impurity functions =====================
def gini(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    p = counts / np.where(total == 0, 1, total)
    return 1.0 - (p ** 2).sum(axis=-1)


def entropy(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    p = counts / np.where(total == 0, 1, total)
    logp = np.log2(np.where(p > 0, p, 1.0))
    return -(p * logp).sum(axis=-1)


# ===================== the tree =====================
class DecisionTree:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 max_features=None, rng=None):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng
        self.impurity = gini if criterion == "gini" else entropy

    # ---------- training ----------
    def fit(self, X, y, sample_weight=None):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=int)
        n = len(self.y_)
        if sample_weight is None:
            self.w_ = np.full(n, 1.0 / n)
        else:
            self.w_ = np.asarray(sample_weight, dtype=float)
        self.n_classes_ = int(self.y_.max()) + 1
        if self.max_features is not None and self.rng is None:
            self.rng = np.random.RandomState(42)

        self.feature_, self.threshold_ = [], []
        self.left_, self.right_ = [], []
        self.value_, self.gain_ = [], []

        self._build(np.arange(n), depth=0)

        self.feature_ = np.array(self.feature_, dtype=int)
        self.threshold_ = np.array(self.threshold_, dtype=float)
        self.left_ = np.array(self.left_, dtype=int)
        self.right_ = np.array(self.right_, dtype=int)
        self.value_ = np.array(self.value_, dtype=int)
        self.gain_ = np.array(self.gain_, dtype=float)
        del self.X_, self.y_, self.w_
        return self

    def _build(self, idx, depth):
        node = len(self.feature_)
        y, w = self.y_[idx], self.w_[idx]
        class_w = np.bincount(y, weights=w, minlength=self.n_classes_)

        # create this node as a leaf first
        self.feature_.append(-1)
        self.threshold_.append(0.0)
        self.left_.append(-1)
        self.right_.append(-1)
        self.value_.append(int(np.argmax(class_w)))
        self.gain_.append(0.0)

        # stopping rules
        if (len(np.unique(y)) <= 1
                or (self.max_depth is not None and depth >= self.max_depth)
                or len(idx) < self.min_samples_split):
            return node

        f, t, g = self._best_split(idx, class_w)
        if f < 0:                      # no split with positive gain
            return node

        go_left = self.X_[idx, f] <= t
        if go_left.all() or not go_left.any():
            return node                # safety: split must actually divide

        # turn the leaf into a split node and build the children
        self.feature_[node], self.threshold_[node], self.gain_[node] = f, t, g
        self.left_[node] = self._build(idx[go_left], depth + 1)
        self.right_[node] = self._build(idx[~go_left], depth + 1)
        return node

    def _best_split(self, idx, class_w):
        X, y, w = self.X_[idx], self.y_[idx], self.w_[idx]
        n, d = X.shape
        W = class_w.sum()
        parent_imp = self.impurity(class_w)

        if self.max_features is None:
            features = np.arange(d)
        else:
            features = np.sort(self.rng.choice(d, self.max_features, replace=False))

        onehot = np.zeros((n, self.n_classes_))
        onehot[np.arange(n), y] = w

        best_f, best_t, best_g = -1, 0.0, 0.0
        for f in features:                         # lower feature index first
            order = np.argsort(X[:, f], kind="stable")
            xs = X[order, f]
            valid = xs[:-1] < xs[1:]               # only cut between different values
            if not valid.any():
                continue
            left = np.cumsum(onehot[order], axis=0)[:-1][valid]
            right = class_w - left
            WL = left.sum(axis=1)
            WR = W - WL
            gains = (parent_imp
                     - (WL / W) * self.impurity(left)
                     - (WR / W) * self.impurity(right))
            g_max = gains.max()
            if g_max > best_g + TOL:               # strictly better (so ties keep lower feature)
                j = np.flatnonzero(gains >= g_max - TOL)[0]   # lowest threshold among ties
                best_f = int(f)
                best_t = (xs[:-1][valid][j] + xs[1:][valid][j]) / 2.0   # midpoint
                best_g = g_max
        return best_f, best_t, best_g

    # ---------- prediction ----------
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        node = np.zeros(len(X), dtype=int)         # everyone starts at the root
        while True:
            f = self.feature_[node]
            active = np.flatnonzero(f >= 0)        # samples not yet at a leaf
            if len(active) == 0:
                break
            n_act = node[active]
            go_left = X[active, f[active]] <= self.threshold_[n_act]
            node[active] = np.where(go_left, self.left_[n_act], self.right_[n_act])
        return self.value_[node]


# ===================== sanity checks =====================
def run_sanity_checks():
    # (a)
    assert np.isclose(gini([3, 3]), 0.5) and np.isclose(entropy([3, 3]), 1.0)
    print("PASS (a) G=0.5 and H=1 bit for counts (3,3)")

    # (b)
    Xb, yb = [[1], [2], [3], [4]], [0, 0, 1, 1]
    t = DecisionTree("entropy").fit(Xb, yb)
    assert len(t.feature_) == 3
    assert np.isclose(t.threshold_[0], 2.5) and np.isclose(t.gain_[0], 1.0)
    assert accuracy(yb, t.predict(Xb)) == 1.0
    print("PASS (b) one split at t=2.5, information gain 1, train accuracy 1.0")

    # (c)
    t = DecisionTree("gini").fit([[5], [5], [5]], [0, 0, 1], sample_weight=[0.1, 0.1, 0.8])
    assert len(t.feature_) == 1 and t.predict([[5]])[0] == 1
    print("PASS (c) identical samples with weights [0.1,0.1,0.8] -> leaf predicting 1")

    # (d)
    Xd = np.random.RandomState(0).rand(5, 3)
    t = DecisionTree("entropy").fit(Xd, [4, 4, 4, 4, 4])
    assert len(t.feature_) == 1 and (t.predict(Xd) == 4).all()
    print("PASS (d) single-class data -> one leaf, no crash")


# ===================== main run =====================
def main():
    run_sanity_checks()

    X, ys, yf, species_names, family_names = load_data()
    train, test = stratified_split(ys)
    Xtr, Xte, ytr, yte = X[train], X[test], ys[train], ys[test]
    fam_map = species_to_family_map(ytr, yf[train])

    print("\n=== Single Decision Tree, unconstrained (results matrix) ===")
    for crit in ("gini", "entropy"):
        start = time.perf_counter()
        tree = DecisionTree(crit, max_depth=None, min_samples_split=2).fit(Xtr, ytr)
        secs = time.perf_counter() - start
        tr = accuracy(ytr, tree.predict(Xtr))
        te_pred = tree.predict(Xte)
        te = accuracy(yte, te_pred)
        fam = accuracy(yf[test], fam_map[te_pred])
        print(f"{crit:8s} nodes={len(tree.feature_):4d} fit={secs:5.2f}s  "
              f"train={100*tr:6.2f}%  test={100*te:6.2f}%  family(test)={100*fam:6.2f}%")
        print(f"RESULT|Single Decision Tree|{crit}|{100*tr:.2f}|{100*te:.2f}|{100*fam:.2f}")

    print("\n=== Depth sweep (not in matrix) ===")
    for crit in ("gini", "entropy"):
        for depth in (1, 2, 3, 5, 8, 12, None):
            tree = DecisionTree(crit, max_depth=depth).fit(Xtr, ytr)
            tr = accuracy(ytr, tree.predict(Xtr))
            te = accuracy(yte, tree.predict(Xte))
            print(f"{crit:8s} max_depth={str(depth):5s} train={100*tr:6.2f}%  test={100*te:6.2f}%")

    print("\n=== max_features check (needed by Part 3) ===")
    tree = DecisionTree("gini", max_features=4, rng=np.random.RandomState(42)).fit(Xtr, ytr)
    print(f"gini, 4 random features per split: train={100*accuracy(ytr, tree.predict(Xtr)):.2f}%  "
          f"test={100*accuracy(yte, tree.predict(Xte)):.2f}%")


if __name__ == "__main__":
    main()