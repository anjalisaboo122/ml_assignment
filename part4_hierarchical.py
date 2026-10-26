import numpy as np
from common import load_data, stratified_split, accuracy, species_to_family_map, confusion_matrix
from part1_decision_tree import DecisionTree


# Hierarchical Classifier 
class HierarchicalClassifier:
    """Router-plus-experts hierarchical classifier.

    Parameters
    ----------
    router  : trained DecisionTree that predicts Family (int 0..3)
    experts : dict {family_int -> trained DecisionTree predicting Species}
    """

    def __init__(self, router, experts):
        self.router = router
        self.experts = experts

    def predict_family(self, X):
        """Return router predictions (family integers) for every row in X."""
        return self.router.predict(np.asarray(X, dtype=float))

    def predict(self, X):
        """Route all samples, call each expert once on its group, return species
        predictions in the original row order (no per-sample Python loops over
        the models)."""
        X = np.asarray(X, dtype=float)
        n = len(X)
        predicted_family = self.predict_family(X)
        predictions = np.empty(n, dtype=int)

        for fam_idx, expert in self.experts.items():
            mask = predicted_family == fam_idx
            if mask.any():
                predictions[mask] = expert.predict(X[mask])

        return predictions


# ===================== Training helpers =====================
def train_hierarchical(Xtr, ytr_species, ytr_family, criterion):
    """Train and return a HierarchicalClassifier.

    Router is trained on all training samples (predict Family).
    Each expert is trained on the TRUE-family subset (predict Species).
    Single-species families produce a constant-predictor leaf — the
    existing DecisionTree handles this correctly (sanity check d).
    """
    # 1. Router
    router = DecisionTree(criterion, max_depth=None, min_samples_split=2)
    router.fit(Xtr, ytr_family)

    # 2. Experts — one per family, trained on true family subsets
    experts = {}
    for fam_idx in np.unique(ytr_family):
        mask = ytr_family == fam_idx
        expert = DecisionTree(criterion, max_depth=None, min_samples_split=2)
        expert.fit(Xtr[mask], ytr_species[mask])
        experts[int(fam_idx)] = expert

    return HierarchicalClassifier(router, experts)


def oracle_predict(X, yf_true, experts):
    """Predict species by routing with the TRUE family (perfect router).
    This gives the oracle accuracy — experts alone, no router errors."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    predictions = np.empty(n, dtype=int)
    for fam_idx, expert in experts.items():
        mask = yf_true == fam_idx
        if mask.any():
            predictions[mask] = expert.predict(X[mask])
    return predictions


# main run 
def main():
    X, ys, yf, species_names, family_names = load_data()
    train, test = stratified_split(ys)
    Xtr, Xte = X[train], X[test]
    ytr_s, yte_s = ys[train], ys[test]
    ytr_f, yte_f = yf[train], yf[test]

    n_species = len(species_names)   # 10
    n_families = len(family_names)   # 4

    # flat tree info for comparison (same split, criterion applied per loop)
    fam_map = species_to_family_map(ytr_s, ytr_f)

    for criterion in ("gini", "entropy"):
        print(f"\n{'='*65}")
        print(f"  Criterion: {criterion}")
        print(f"{'='*65}")

        clf = train_hierarchical(Xtr, ytr_s, ytr_f, criterion)

        # Router accuracy 
        router_tr_pred = clf.predict_family(Xtr)
        router_te_pred = clf.predict_family(Xte)
        router_tr = accuracy(ytr_f, router_tr_pred)
        router_te = accuracy(yte_f, router_te_pred)
        print(f"Router accuracy:   train={100*router_tr:6.2f}%  test={100*router_te:6.2f}%")

        # Overall (species) accuracy 
        overall_tr_pred = clf.predict(Xtr)
        overall_te_pred = clf.predict(Xte)
        overall_tr = accuracy(ytr_s, overall_tr_pred)
        overall_te = accuracy(yte_s, overall_te_pred)
        print(f"Overall accuracy:  train={100*overall_tr:6.2f}%  test={100*overall_te:6.2f}%")

        #  Oracle accuracy (perfect router -> experts only) 
        oracle_pred = oracle_predict(Xte, yte_f, clf.experts)
        oracle_acc = accuracy(yte_s, oracle_pred)
        print(f"Oracle accuracy:   test ={100*oracle_acc:6.2f}%")

        # Failure fractions 
        # pA: router correct but expert wrong
        router_correct = (router_te_pred == yte_f)
        expert_wrong   = (overall_te_pred != yte_s)
        pA = float(np.mean(router_correct & expert_wrong))
        # pB: router wrong (species always wrong in this case)
        pB = 1.0 - router_te
        print(f"Failure fractions: pA (router right, expert wrong)={100*pA:.2f}%  "
              f"pB (router wrong)={100*pB:.2f}%")

        # Numerical check: Aoverall = Arouter - pA 
        overall_check = router_te - pA
        match = np.isclose(overall_te, overall_check, atol=1e-9)
        print(f"Check Aoverall = Arouter - pA: "
              f"{100*overall_te:.4f}% = {100*router_te:.4f}% - {100*pA:.4f}% "
              f"= {100*overall_check:.4f}%  [{'PASS' if match else 'FAIL'}]")

        #  4x4 Router confusion matrix 
        cm_router = confusion_matrix(yte_f, router_te_pred, n_families)
        col_w = max(len(n) for n in family_names) + 2
        print(f"\nRouter 4x4 confusion matrix (rows=true, cols=predicted):")
        header = "".join(f"{n:>{col_w}}" for n in family_names)
        print(f"{'':>{col_w}}  {header}")
        for i, row in enumerate(cm_router):
            vals = "".join(f"{v:>{col_w}}" for v in row)
            print(f"{family_names[i]:>{col_w}}  {vals}")

        #  10x10 Species confusion matrix 
        cm_species = confusion_matrix(yte_s, overall_te_pred, n_species)
        sp_w = max(len(n) for n in species_names) + 1
        print(f"\nSpecies 10x10 confusion matrix (rows=true, cols=predicted):")
        sp_header = "".join(f"{n[:8]:>9}" for n in species_names)
        print(f"{'':>{sp_w}}  {sp_header}")
        for i, row in enumerate(cm_species):
            vals = "".join(f"{v:>9}" for v in row)
            print(f"{species_names[i]:>{sp_w}}  {vals}")

        # Per-family test accuracy 
        print(f"\nPer-family test accuracy:")
        for fam_idx, fname in enumerate(family_names):
            mask = yte_f == fam_idx
            if mask.any():
                fam_acc = accuracy(yte_s[mask], overall_te_pred[mask])
                print(f"  {fname}: {100*fam_acc:.2f}%")

        # Comparison with flat model (Part 1) 
        flat_tree = DecisionTree(criterion, max_depth=None, min_samples_split=2)
        flat_tree.fit(Xtr, ytr_s)
        flat_te_pred = flat_tree.predict(Xte)
        flat_species_acc = accuracy(yte_s, flat_te_pred)
        flat_family_acc  = accuracy(yte_f, fam_map[flat_te_pred])
        print(f"\nComparison with flat tree ({criterion}):")
        print(f"  Species accuracy — flat: {100*flat_species_acc:.2f}%  "
              f"hierarchical: {100*overall_te:.2f}%")
        print(f"  Family  accuracy — flat derived: {100*flat_family_acc:.2f}%  "
              f"router: {100*router_te:.2f}%")
        bottleneck = "router" if pB > pA else "experts"
        print(f"  Bottleneck: {bottleneck} "
              f"(pA={100*pA:.2f}%  pB={100*pB:.2f}%)")
        hierarchy_helps = overall_te > flat_species_acc
        print(f"  Does hierarchy help? {'Yes' if hierarchy_helps else 'No'} "
              f"({100*overall_te:.2f}% vs {100*flat_species_acc:.2f}%)")

        # machine-parsable RESULT lines
        # Family acc. column = router accuracy (per assignment spec)
        print(f"RESULT|Hierarchical Decision Tree|{criterion}|"
              f"{100*overall_tr:.2f}|{100*overall_te:.2f}|{100*router_te:.2f}")
        # supplementary table 3
        print(f"RESULT_SUPP|{criterion}|router={100*router_te:.2f}|"
              f"oracle={100*oracle_acc:.2f}|pA={100*pA:.2f}|pB={100*pB:.2f}")


if __name__ == "__main__":
    main()
