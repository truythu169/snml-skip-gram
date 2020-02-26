# Applying information criteria in Skip-gram dimensionality selection

The detail about installation will be soon.

## 1. Preprocess data
Run prepocess.py file to prepocess data. This file takes .txt file as input.
Please remove special characters such as .,:? etc in the text file.

## 2. Train Skip-gram
Run tf_based/train.py to train Skip-gram model

## 3. Estimate AIC & BIC
Model class in bic/model.py provide both method to estimate AIC and BIC.

## 4. Estimate SNML codelength
See snml/tf_based/train_snml.py for more detail about the implementation.

## 5. Artificial data
See notebooks/Generate context distributions - word analogy.ipynb for the detail about generating artificial data. 