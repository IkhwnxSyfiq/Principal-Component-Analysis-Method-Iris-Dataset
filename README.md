Lab Assignment: Principal Component Analysis (PCA) on the Iris Dataset

This repository contains the code and report for a university lab assignment on Principal Component Analysis (PCA), a fundamental technique for dimensionality reduction in Big Data Analytics. The project demonstrates a step-by-step implementation of PCA on the classic Iris dataset, from data preprocessing to visualization.

📄 Assignment Overview
This lab was completed as part of the BEB43403 - Big Data Analytics course at Universiti Kuala Lumpur (UniKL British Malaysian Institute). The objective was to gain a hands-on understanding of PCA, its mathematical underpinnings, and its application for visualizing high-dimensional data.

👥 Authors
Name	Student ID
Muhammad Ikhwan Syafiq bin Norsham	51221221125
Muhammad Waiz bin Nor Kamal	51221221053
🧪 Project Structure
The lab is structured into three main parts:

Part B: Understanding PCA - Theoretical questions on PCA concepts.

Part C: PCA Lab Tutorial - A practical, step-by-step coding guide to implementing PCA from scratch and using scikit-learn.

Full Code - The complete, executable Python script.

📊 Dataset
The Iris dataset is a classic dataset in machine learning and statistics. It contains 150 samples from three species of Iris flowers (Iris setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the lengths and widths of the sepals and petals.

Dataset File: iris.csv

Samples: 150

Features: 4 (sepal.length, sepal.width, petal.length, petal.width)

Target Variable: variety (Species name)

🔧 Implementation Steps
The code follows a meticulous process to perform PCA:

Data Loading & Inspection: The dataset is loaded using pandas, and its structure is examined.

Data Preprocessing: The features are standardized using StandardScaler to have a mean of 0 and a standard deviation of 1.

Covariance Matrix Calculation: The covariance matrix of the standardized data is computed using numpy.cov.

Eigendecomposition: Eigenvalues and eigenvectors are calculated from the covariance matrix using numpy.linalg.eig. These define the principal components.

Dimensionality Reduction: Using scikit-learn's PCA tool, the data is projected onto the first two principal components.

Visualization: The transformed data is plotted in a 2D scatter plot, color-coded by Iris species, to visualize the separation achieved by PCA.

📈 Results & Analysis
The PCA transformation successfully reduced the 4-dimensional Iris data into 2 dimensions while preserving the majority of the variance in the data.

Explained Variance: The first two principal components explain a high percentage of the total variance.

Cluster Separation: The resulting scatter plot shows clear clusters:

Iris-setosa is completely separated from the other two species.

Iris-versicolor and Iris-virginica show some overlap but remain largely distinct, indicating that two dimensions are sufficient to capture the primary differences between species.

This visualization demonstrates how PCA can be used for exploratory data analysis and as a preprocessing step for other machine learning algorithms.


🤖 Other Big Data Techniques
As discussed in the report, other powerful techniques for big data analytics include:

K-Means Clustering: An unsupervised learning algorithm for partitioning data into distinct clusters based on similarity.

Apache Spark MLlib: A scalable machine learning library designed for distributed data processing on big data clusters.

✅ Conclusion
This lab provided a comprehensive walkthrough of PCA, reinforcing key concepts in dimensionality reduction. By implementing it both manually (via covariance and eigendecomposition) and with high-level libraries, we gained insight into the mechanics and practical utility of PCA for simplifying complex datasets and revealing intrinsic patterns.

📚 References
pandas Documentation

scikit-learn Documentation

numpy Documentation

matplotlib Documentation
