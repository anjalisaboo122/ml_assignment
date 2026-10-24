from part1_decision_tree import gini, entropy

print(gini([3, 3]), entropy([3, 3]))          # sanity check (a)
print(gini([6, 0]), entropy([6, 0]))
print(gini([2, 4]), entropy([2, 4]))
print(gini([[2, 0], [1, 2], [0, 2]]))
print(entropy([[1, 0], [1, 2], [0, 1]]))