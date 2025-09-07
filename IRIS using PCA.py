#calling for python libraries
import pandas as pd #provides data structures like DataFrame
from sklearn.preprocessing import StandardScaler #provides simple and efficient tools for data analysis and modeling
import numpy as np #provides multi-dimensional arrays and matrices
from sklearn.decomposition import PCA # to transform the original features of a dataset into a new set of uncorrelated features, called principal components
import matplotlib.pyplot as plt #provides a variety of functions to create different types of plots, charts, and visualizations
from sklearn.preprocessing import LabelEncoder #provides several functions for data preprocessing and feature engineering

# Load the dataset
iris_data = pd.read_csv('/content/iris.csv')

# Understanding the data set
numSamples, numFeatures = iris_data.shape #used to extract the number of samples (rows) and the number of features (columns)
print("Number of samples:", numSamples)
print("Number of features:", numFeatures)
#Target column is present in the dataset, you can print the unique values in the target column which is my tar
print("Target names:", iris_data['variety'].unique())

# Display the size of the dataset (rows, columns)
data_size = iris_data.shape

# Display the number of parameters/features and their details
parameter_size = iris_data.shape[1]  # Number of columns
parameter_details = iris_data.columns.tolist()  # List of column names

# Print the results
print("Data size:", data_size)
print("Parameter / feature size:", parameter_size)
print("Parameter / feature details:", parameter_details)

# Display the entire data
print(iris_data)

x = iris_data.iloc[:, :-1]  # Features (all columns except the last one)
y = iris_data.iloc[:, -1]   # Labels (the last column)

# Displaying x and y
print("X (Features):")
print(x)

print("\nY (Labels):")
print(y)

scaler = StandardScaler() #preprocessing technique commonly used in machine learning to standardize the features of a dataset.
scaled_data = scaler.fit_transform(x) #The fit method is used to compute the mean and standard deviation of each feature in the dataset

# Displaying the scaled data
scaled_data_df = pd.DataFrame(scaled_data, columns=x.columns)
print("Scaled Data:")
print(scaled_data_df)

# Calculate the mean from the scaled data
scaled_data_mean = np.mean(scaled_data, axis=0)

# Displaying the mean results based on the StandardScaler
print("Mean Result based on StandardScaler:")
print(scaled_data_mean)

# Subtract the scaled data mean from each row in the original data
new_data = x - scaled_data_mean

# Displaying the new_data
print("New Data after subtracting with the scaled data mean:")
print(new_data)

# new_data is the modified data obtained by subtracting scaled_data_mean
covariance_matrix = np.cov(new_data, rowvar=False)

# Displaying the covariance matrix
print("Covariance Matrix:")
print(covariance_matrix)

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Displaying the eigenvectors and eigenvalues
print("Eigenvectors:")
print(eigenvectors)
print("\nEigenvalues:")
print(eigenvalues)

# Initialize PCA with the number of components you want to retain
# For example, if you want to retain 2 components, set n_components=2
pca = PCA(n_components=2)

# Fit and transform the data using PCA
pca_result = pca.fit_transform(new_data)

# Display the explained variance ratio
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Display the transformed data
pca_df = pd.DataFrame(data=pca_result, columns=['Principal Component 1', 'Principal Component 2'])
print("\nTransformed Data:")
print(pca_df)

# Concatenate the transformed data with the target variable
pca_df_with_target = pd.concat([pca_df, iris_data['variety']], axis=1)

# Create scatter plot
plt.figure(figsize=(8, 6))
for variety in iris_data['variety'].unique():
    subset = pca_df_with_target[pca_df_with_target['variety'] == variety]
    plt.scatter(subset['Principal Component 1'], subset['Principal Component 2'], label=variety)

# Add labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Results using Standardized Data')
plt.legend()
plt.show()
