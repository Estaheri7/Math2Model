# Sampling Notebook

This notebook provides a set of tools for generating different types of sample data using Python and NumPy. The `Sample` class allows you to create uniform, normal, and classification samples, and also provides functionality to add a bias term to the data.

## Class and Methods

### Sample Class

The `Sample` class is designed to generate sample data with specified dimensions and sample size. It includes the following methods:

- `__init__(self, n_samples, n_dimension)`: Initializes the class with the number of samples and dimensions.
- `create_uniform_samples(self, low=0, high=1)`: Generates uniform samples within the specified range.
- `create_normal_samples(self, mean=0, standard_deviation=1)`: Generates normal (Gaussian) samples with the specified mean and standard deviation.
- `create_classification_samples(self, type='normal', l=0, h=1, split_rate=0.5, shuffle=False)`: Generates classification samples, either normal or uniform, and labels them based on the split rate.
- `add_bias_to_data(self)`: Adds a bias term (column of ones) to the data.

## Explanation of Algorithms

### Uniform Sampling
Uniform sampling generates data points that are uniformly distributed within a specified range. This means each value within the range has an equal probability of being selected.

### Normal Sampling
Normal sampling generates data points that follow a Gaussian distribution, characterized by a specified mean and standard deviation. This distribution is also known as the bell curve.

### Classification Sampling
Classification sampling generates data points for classification tasks. It can create either normal or uniform samples and labels them based on a specified split rate. The data can also be shuffled to ensure randomness.

### Adding Bias
Adding a bias term involves appending a column of ones to the data matrix. This is often used in machine learning algorithms to account for the intercept term in linear models.

This notebook provides a flexible and easy-to-use framework for generating sample data, which can be useful for testing and developing machine learning algorithms.