#!/usr/bin/env python
# coding: utf-8

# In[70]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# List of folder paths
folder_paths = [

    "/Users/zjz/Desktop/D/projects/Gaming/data/1E/costdiff",
    "/Users/zjz/Desktop/D/projects/Gaming/data/2E/costdiff",
    "/Users/zjz/Desktop/D/projects/Gaming/data/3E/costdiff",
    "/Users/zjz/Desktop/D/projects/Gaming/data/4E/costdiff"
]

# Function to read and process each curve
def read_curve(file_path, num_points=100):
    # Load the file and infer the header dynamically
    df = pd.read_csv(file_path)

    # Find the first numerical column as 'Cost_Difference' equivalent
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        raise ValueError(f"No numerical columns found in {file_path}")
    distance = df[numeric_columns[0]].values  # Use the first numerical column

    # Generate a uniform time vector assuming each row is a sequential time point
    time = np.arange(len(distance))

    # Interpolating to ensure all curves have the same length
    interp_func = np.interp(np.linspace(0, len(time)-1, num_points), time, distance)

    return interp_func

# Read and process all valid CSV files in the folders
all_curves = []
for folder_path in folder_paths:
    for file_name in os.listdir(folder_path):
        # Skip files whose names start with a number
        if file_name[0].isdigit() and file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                try:
                    curve = read_curve(file_path)
                    all_curves.append(curve)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

# Convert the list of curves into a NumPy array
data_matrix = np.array(all_curves)

# Check for NaN or infinite values and remove invalid curves
valid_curves = []
for curve in data_matrix:
    if not np.any(np.isnan(curve)) and not np.any(np.isinf(curve)):
        valid_curves.append(curve)

# Convert the valid curves back to a NumPy array
data_matrix = np.array(valid_curves)

# Ensure data_matrix is not empty
if data_matrix.size == 0:
    raise ValueError("No valid curves remaining after filtering NaN or infinite values.")

# Polynomial basis expansion function
def polynomial_basis_expansion(data, degree=3):
    """
    Expands the input data using polynomial basis up to a specified degree.
    """
    num_points = data.shape[1]
    X = np.linspace(0, 1, num_points).reshape(-1, 1)  # Normalized time points
    poly_features = np.hstack([X**i for i in range(degree + 1)])  # Polynomial features
    coefficients = []

    for curve in data:
        model = LinearRegression()
        model.fit(poly_features, curve)
        coefficients.append(model.coef_)

    return np.array(coefficients), poly_features

# Perform polynomial basis expansion
degree = 10  # 
coefficients, poly_features = polynomial_basis_expansion(data_matrix, degree)

# Visualize the reconstructed curves and compare them to the original curves
plt.figure(figsize=(12, 6))
for i, curve in enumerate(data_matrix[:5]):  # Visualize the first 5 curves
    reconstructed_curve = poly_features @ coefficients[i]
    plt.plot(curve, label=f"Original Curve {i+1}")
    plt.plot(reconstructed_curve, '--', label=f"Reconstructed Curve {i+1}")
plt.xlabel('Time')
plt.ylabel('Cost Difference')
plt.title('Original vs Reconstructed Curves (Polynomial Basis Expansion)')
plt.legend()
plt.show()

# Optional: Visualize the coefficients
plt.figure(figsize=(10, 6))
for i in range(degree + 1):
    plt.plot(coefficients[:, i], label=f"Coefficient for x^{i}")
plt.xlabel('Curve Index')
plt.ylabel('Coefficient Value')
plt.title('Polynomial Coefficients Across Curves')
plt.legend()
plt.show()






# In[71]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 确定聚类数目
k = 4  # 假设分成 4 类
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(coefficients)

# 打印每条曲线的聚类标签
print(f"Cluster labels: {labels}")

from sklearn.decomposition import PCA

# 计算轮廓系数
sil_score = silhouette_score(coefficients, labels)
print(f"Silhouette Score: {sil_score}")

# 使用 PCA 将多项式系数降维到 2D
pca = PCA(n_components=2)
coeff_2d = pca.fit_transform(coefficients)

# 可视化聚类结果
plt.figure(figsize=(8, 6))
for cluster in range(k):
    cluster_points = coeff_2d[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Cluster Visualization (PCA)')
plt.legend()
plt.show()



# In[72]:



clusters = {i: [] for i in range(k)}


for curve, label in zip(data_matrix, labels):
    clusters[label].append(curve)


mean_curves = {cluster: np.mean(curves, axis=0) for cluster, curves in clusters.items()}


plt.figure(figsize=(10, 6))
for cluster, mean_curve in mean_curves.items():
    plt.plot(mean_curve, label=f"Cluster {cluster} Average", linewidth=2)

plt.xlabel('Time')
plt.ylabel('Cost Difference')
plt.title('Average Curves of Each Cluster')
plt.legend()
plt.show()



# In[ ]:


## Stablity Check
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Set parameters
num_points = 100  # number of interpolated points per curve
segment_K = 10     # number of segments per curve
poly_degree = 3   # polynomial degree for each segment

# Folder paths
folder_paths = [
    "/Users/zjz/Desktop/D/projects/Gaming/data/1E/costdiff",
    "/Users/zjz/Desktop/D/projects/Gaming/data/2E/costdiff",
    "/Users/zjz/Desktop/D/projects/Gaming/data/3E/costdiff",
    "/Users/zjz/Desktop/D/projects/Gaming/data/4E/costdiff"
]

# Function to read and interpolate curve
def read_curve(file_path, num_points=100):
    df = pd.read_csv(file_path)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        raise ValueError(f"No numerical columns found in {file_path}")
    distance = df[numeric_columns[0]].values
    time = np.arange(len(distance))
    interp_curve = np.interp(np.linspace(0, len(time)-1, num_points), time, distance)
    return interp_curve

# Read and clean all curves
all_curves = []
for folder_path in folder_paths:
    for file_name in os.listdir(folder_path):
        if file_name[0].isdigit() and file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            try:
                curve = read_curve(file_path, num_points=num_points)

                if curve[0] <= 400 and not np.any(np.isnan(curve)) and not np.any(np.isinf(curve)):
                    all_curves.append(curve)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

data_matrix = np.array(all_curves)
if data_matrix.size == 0:
    raise ValueError("No valid curves remaining.")

# === Function: Segment-wise polynomial fitting ===
def segment_constant_coefficients(curves, degree=3, K=4):
    """
    Fit polynomial to each segment and return only the constant term (coef_[0]) from each segment.
    Resulting in K coefficients per curve.
    """
    n_curves, n_points = curves.shape
    segment_length = n_points // K
    all_consts = []

    for curve in curves:
        const_terms = []
        for k in range(K):
            start = k * segment_length
            end = (k + 1) * segment_length if k < K - 1 else n_points
            segment = curve[start:end]
            X = np.linspace(0, 1, end - start).reshape(-1, 1)
            poly_features = np.hstack([X**d for d in range(degree + 1)])
            model = LinearRegression()
            model.fit(poly_features, segment)
            const_terms.append(model.coef_[1])  # Only take constant term
        all_consts.append(const_terms)

    return np.array(all_consts)

# Fit all curves segment-wise

coefficients = segment_constant_coefficients(data_matrix, degree=poly_degree, K=segment_K)

# KMeans  + PCA  + silhouette
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(coefficients)
print(f"Cluster labels: {labels}")



clusters = {i: [] for i in range(k)}


for curve, label in zip(data_matrix, labels):
    clusters[label].append(curve)


mean_curves = {cluster: np.mean(curves, axis=0) for cluster, curves in clusters.items()}


plt.figure(figsize=(10, 6))
for cluster, mean_curve in mean_curves.items():
    plt.plot(mean_curve, label=f"Cluster {cluster} Average", linewidth=2)

plt.xlabel('Time')
plt.ylabel('Cost Difference')
plt.title('Average Curves of Each Cluster')
plt.legend()
plt.show()


