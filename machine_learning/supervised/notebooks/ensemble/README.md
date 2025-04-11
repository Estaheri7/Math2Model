# Ensemble Learning Algorithms

This directory contains Jupyter notebooks that demonstrate various ensemble learning algorithms used in supervised machine learning.

## Overview

Ensemble learning is a technique that combines multiple base models to produce a stronger overall model. The main idea is that by combining the predictions of several models, the ensemble model can achieve better performance and generalization than any individual model.

## Algorithms Covered (AdaBoost for now)

1. **Bagging (Bootstrap Aggregating)**
    - Bagging involves training multiple instances of the same algorithm on different subsets of the training data, and then averaging their predictions. Random Forest is a popular example of a bagging algorithm.

2. **Boosting**
    - Boosting trains models sequentially, with each new model focusing on the errors made by the previous ones. The final model is a weighted sum of all the models. Examples include AdaBoost, Gradient Boosting, and XGBoost.

3. **Stacking**
    - Stacking involves training multiple base models and then using another model (meta-model) to combine their predictions. The base models are trained on the original dataset, while the meta-model is trained on the predictions of the base models.
