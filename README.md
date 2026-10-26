BITS F464: Machine Learning  |  Assignment 1
============================================

Team members
------------
  Jayaditya Kushwah - 2024ADPS0188H
  Aadi M. Ajit Kumar - 2024A7PS2014H
  Anjali Saboo - 2024A7PS2011H

Environment
-----------
  Python  : 3.14.5
  numpy   : 2.5.4
  pandas  : 3.1.0
  Package manager: uv

  Allowed libraries only: numpy, pandas (matplotlib for optional plots).
  No sklearn, scipy, torch, tensorflow, statsmodels, xgboost, or lightgbm.

Dataset
-------
  Anuran Calls (MFCCs), UCI ML Repository
  File   : Frogs_MFCCs.csv  (place in the same directory as the scripts)
  Records: 7,195  |  Features: MFCCs_1 … MFCCs_22  |  Labels: Family (4), Species (10)

Commands to run
---------------
  From the directory containing the scripts and Frogs_MFCCs.csv:

    uv run python part1_decision_tree.py
    uv run python part2_bagging_boosting.py
    uv run python part3_randomforests.py
    uv run python part4_hierarchical.py
    uv run python part5_naive_bayes.py

  To generate results_matrix.pdf (run from the A1 root, one level above ml_assignment/):
    uv run --project ml_assignment python generate_results.py

  Output format: every script emits machine-parsable lines of the form
    RESULT|method|criterion|train%|test%|family%
  which generate_results.py collects to build the PDF table automatically.

Written answers to assignment questions
-----------------------------------------

Q (Part 2): Why do decision stumps (max_depth=1) perform differently from
            depth-3 trees in AdaBoost?

  A depth-1 stump can only split on a single feature, so each weak learner
  corrects a relatively coarse partition of the error. Boosting compensates
  by keeping more rounds, but stumps often plateau early because no single
  feature suffices to separate the 10-class frog dataset without combining
  multiple splits. Depth-3 trees have enough capacity to capture two or
  three compound conditions per round, making each alpha contribution more
  informative. With SAMME on a 10-class problem the effective "random
  baseline" is 10%, so stumps must beat a higher relative threshold before
  they accumulate useful alphas, and rounds that fail to do so are discarded.

Q (Part 5): How does the correlation between MFCC features affect Gaussian
            Naive Bayes?

  MFCCs are computed from adjacent DCT basis functions of the same signal,
  so neighbouring coefficients share information and are moderately
  correlated. Naive Bayes treats the 22 features as conditionally
  independent given the class; when features are correlated, the model
  effectively counts the shared evidence multiple times and produces
  overconfident (poorly calibrated) posteriors. Despite this, the argmax
  decision can still be competitive, because the ranking across classes is
  often preserved even when absolute probabilities are wrong — a well-known
  empirical property of Naive Bayes. Accuracy on a held-out set will reveal
  whether the independence violation is benign or harmful for this dataset.

Q (Part 4): Is the router or the experts the bottleneck?
            Does the hierarchy help on this dataset?

  See the printed output of part4_hierarchical.py: the script computes pA
  (router correct, expert wrong) and pB (router wrong = 1 - router accuracy)
  and explicitly labels the bottleneck. In general for this dataset the
  family boundaries are well-separated in MFCC space, so the router tends to
  be highly accurate; any remaining error comes from the within-family expert.
  Whether the hierarchy improves on the flat tree depends on the criterion
  and is reported directly in the script's comparison block.

Leakage note
------------
  Multiple calls from the same recording appear in both train and test sets,
  so reported accuracies are optimistic relative to truly unseen recordings.
  This is a property of the UCI dataset split, not a bug in the code.
