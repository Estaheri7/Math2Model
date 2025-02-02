# Principal Component Analysis (PCA) Implementation

This repository contains a simple implementation of Principal Component Analysis (PCA) from scratch using Python. PCA is a popular dimensionality reduction technique used in machine learning and data analysis to reduce the number of features in a dataset while retaining most of the variance.

## Algorithm Explanation

Principal Component Analysis (PCA) is a statistical procedure that uses orthogonal transformation to convert a set of possibly correlated variables into a set of linearly uncorrelated variables called principal components. The number of principal components is less than or equal to the number of original variables. This transformation is defined in such a way that the first principal component has the largest possible variance, and each succeeding component has the highest variance possible under the constraint that it is orthogonal to the preceding components.

### Steps Involved in PCA:

1. **Standardize the Data**: Subtract the mean of each feature from the dataset to center the data around the origin.
2. **Compute the Covariance Matrix**: Calculate the covariance matrix to understand the relationships between different features.
3. **Compute Eigenvalues and Eigenvectors**: Determine the eigenvalues and eigenvectors of the covariance matrix to identify the principal components.
4. **Sort Eigenvalues and Eigenvectors**: Sort the eigenvalues and their corresponding eigenvectors in descending order.
5. **Select Top k Eigenvectors**: Choose the top k eigenvectors that correspond to the k largest eigenvalues to form the transformation matrix.
6. **Transform the Data**: Multiply the original dataset by the transformation matrix to obtain the reduced dataset.

## Usage

The implementation is provided in a Jupyter Notebook (`PCA.ipynb`). Below is a brief overview of the main components of the code:

- **PCA Class**: A class that encapsulates the PCA algorithm with methods to fit the model and transform the data.
- **fit Method**: Computes the principal components from the input data.
- **transform Method**: Projects the input data onto the principal components.
- **fit_transform Method**: Combines the fit and transform steps for convenience.
