import time
import numpy as np
from common import load_data, stratified_split, accuracy, species_to_family_map
from part1_decision_tree import DecisionTree
from part2_bagging_boosting import Bagging, majority_vote


# Random Forest
class RandomForest:
    """Bagging + a fresh random subset of m features at EVERY split."""
    def __init__(self, n_estimators=50, criterion="gini", max_features=4, seed=42):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features        # m = floor(sqrt(22)) = 4
        self.seed = seed

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        N = len(y)
        self.n_classes_ = int(y.max()) + 1
        rng = np.random.RandomState(self.seed)  # ONE rng: bootstraps AND feature subsets
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.randint(0, N, size=N)      # bootstrap, same as bagging
            tree = DecisionTree(self.criterion, max_depth=None, min_samples_split=2,
                                max_features=self.max_features,
                                rng=rng)          # pass the SAME rng -> trees differ
            self.trees_.append(tree.fit(X[idx], y[idx]))
        return self

    def predict(self, X):
        preds = np.array([t.predict(X) for t in self.trees_])
        return majority_vote(preds, self.n_classes_)


def avg_pairwise_agreement(trees, X):
    """Fraction of rows on which two trees agree, averaged over all pairs of trees."""
    P = np.array([t.predict(X) for t in trees])
    T = len(P)
    total, pairs = 0.0, 0
    for i in range(T):
        for j in range(i + 1, T):
            total += np.mean(P[i] == P[j])
            pairs += 1
    return total / pairs


# main run 
def main():
    X, ys, yf, _, _ = load_data()
    train, test = stratified_split(ys)
    Xtr, Xte, ytr, yte = X[train], X[test], ys[train], ys[test]
    fam_map = species_to_family_map(ytr, yf[train])

    for crit in ("gini", "entropy"):
        print(f"\n=== criterion = {crit} ===")
        t0 = time.perf_counter()
        rf = RandomForest(50, crit, max_features=4).fit(Xtr, ytr)
        secs = time.perf_counter() - t0
        tr = accuracy(ytr, rf.predict(Xtr))
        te_pred = rf.predict(Xte)
        te = accuracy(yte, te_pred)
        fam = accuracy(yf[test], fam_map[te_pred])
        print(f"Random Forest    {crit:8s} fit={secs:5.1f}s  train={100*tr:6.2f}%  "
              f"test={100*te:6.2f}%  family(test)={100*fam:6.2f}%")
        print(f"RESULT|Random Forest|{crit}|{100*tr:.2f}|{100*te:.2f}|{100*fam:.2f}")

        # comparison with single tree (Part 1) and Bagging (Part 2)
        single = DecisionTree(crit).fit(Xtr, ytr)
        bag = Bagging(25, crit).fit(Xtr, ytr)
        print(f"  test accuracy:  single tree={100*accuracy(yte, single.predict(Xte)):6.2f}%  "
              f"bagging={100*accuracy(yte, bag.predict(Xte)):6.2f}%  forest={100*te:6.2f}%")

        # extra (not graded): are forest trees more diverse than bagging trees?
        print(f"  avg pairwise agreement (test):  bagging={avg_pairwise_agreement(bag.trees_, Xte):.4f}  "
              f"forest={avg_pairwise_agreement(rf.trees_, Xte):.4f}")


if __name__ == "__main__":
    main()