# Random Forest from Scratch: Wine Quality Classification

## Overview
This project implements a Random Forest classifier from scratch (without using scikit-learn or other ML libraries) to predict wine quality based on physicochemical properties. The implementation is in Python and demonstrates the core logic of decision trees and ensemble learning.

## Dataset
- **Source:** [UCI Machine Learning Repository - Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **File:** `data/winequality-red.csv`
- **Description:** Each row represents a red wine sample with physicochemical properties (e.g., acidity, sugar, pH) and a quality score (label).

## Project Structure
```
.
├── data/
│   └── winequality-red.csv         # Dataset
├── decision_tree_functions.py      # Decision tree and prediction logic
├── helper_functions.py             # Data splitting, accuracy, and feature type helpers
├── notebook.ipynb                  # Main notebook: data prep, training, evaluation
├── report.pdf                      # Project report (optional)
└── readme.md                       # This file
```

## Setup
1. **Clone the repository**
2. **Install dependencies** (Python 3.7+ recommended):

```bash
pip install numpy pandas
```

## Usage
- **Jupyter Notebook:**
  1. Open `notebook.ipynb` in Jupyter.
  2. Run all cells to:
     - Load and preprocess the data
     - Train a Random Forest classifier (using your own decision tree implementation)
     - Evaluate accuracy on the test set

- **Key Functions:**
  - `decision_tree_functions.py`: Implements decision tree training and prediction
  - `helper_functions.py`: Includes train/test split, accuracy calculation, and feature type detection

## Example Results
- The notebook demonstrates training a Random Forest with 4 trees, each with a max depth of 4, and achieves an accuracy of ~73% on the test set.

## Requirements
- Python 3.7+
- numpy
- pandas

## Notes
- No external machine learning libraries (e.g., scikit-learn) are used for the Random Forest or decision trees.
- The code is intended for educational purposes and can be extended for more advanced use cases.

## License
This project is for educational use. Please cite the UCI dataset if you use the data.
