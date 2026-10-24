from common import load_data, stratified_split, species_to_family_map

X, ys, yf, sn, fn = load_data()
train, test = stratified_split(ys)
print(X.shape, len(train), len(test))
print(sn)
print(fn)
print(species_to_family_map(ys[train], yf[train]))