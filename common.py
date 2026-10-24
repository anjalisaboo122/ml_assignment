import numpy as np
import pandas as pd


def stratified_split(labels, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    test = []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        test.extend(idx[:int(round(test_frac * len(idx)))])
    test = np.sort(np.array(test))
    train = np.setdiff1d(np.arange(len(labels)), test)
    return train, test


def load_data(path="Frogs_MFCCs.csv"):
    df = pd.read_csv(path)

    # 1. feature columns = every column whose name starts with "MFCCs"
    feat_cols = [c for c in df.columns if c.startswith("MFCCs")]

    # 2. X = df[those columns] as a float numpy array -> shape (7195, 22)
    X = df[feat_cols].to_numpy(dtype=float)

    # 3. species text -> integers 0..9 in sorted order
    species_names, y_species = np.unique(df["Species"].to_numpy(), return_inverse=True)

    # 4. same idea for Family -> integers 0..3
    family_names, y_family = np.unique(df["Family"].to_numpy(), return_inverse=True)

    # return X, y_species, y_family, species_names, family_names
    return X, y_species, y_family, species_names, family_names
    


def accuracy(y_true, y_pred):
    # fraction of positions where they're equal (one line, use np.mean)
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    


def species_to_family_map(y_species_train, y_family_train):
    # make an int array of length 10 (one slot per species)
    n_species = y_species_train.max() + 1

    # for each species s: look at training rows with species s,
    # and store their family number in slot s

    mapping = np.zeros(n_species, dtype=int)
    for s in range(n_species):
        mapping[s] = y_family_train[y_species_train == s][0]
    return mapping

def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm