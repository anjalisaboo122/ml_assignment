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
  numpy   : 2.5.2
  pandas  : 3.0.5 

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

    python3 part1_decision_tree.py
    python3 part2_bagging_boosting.py
    python3 part3_random_forest.py
    python3 part4_hierarchical.py
    python3 part5_naive_bayes.py

  To generate results_matrix.pdf (run from the A1 root, one level above ml_assignment/):
    uv run --project ml_assignment python generate_results.py

  Each script prints lines of the form
    RESULT|method|criterion|train%|test%|family%
  which are the numbers in results_matrix.pdf
