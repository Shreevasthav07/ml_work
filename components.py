#importing required Libraries to Perform PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#======================================================

#This is How we create a CSV File Using Pandas
data ={
    "X":[2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1],
    "Y":[2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9]
}
df = pd.DataFrame(data)
csv_path = "pca_data.csv"
df.to_csv(csv_path,index=False)
#========================================================

#Read the data from a CSV
df = pd.read_csv("pca_data.csv")
print("original Data:\n",df,"\n")
#========================================================

#Compute Means of Each Variable
means = df.mean()
print("Means:\n",means,"\n")
#=========================================================

#Standardize (mean-center) the Data
#X_std = X-mean
given_data = df.values #all X,Y Values stored in given data variable here
mean_centered_data = given_data - means.values
print("Mean Centered Data: \n",mean_centered_data,"\n")
#===========================================================================

#Compute Covariance Matrix (features x features)
#cov = (mean_centred_data ^T * mean_centered_data) / (n_samples-1)
# ^T stands for transpose of matrix

cov_matrix = np.cov(mean_centered_data.T)
print("Covariance Matrix:\n", cov_matrix,"\n")
#============================================================================

#Compute eigen_values, eigen_vectors
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
print("Eigenvalues:\n", eigen_values, "\n")
print("Eigenvectors (columns):\n", eigen_vectors, "\n")
#eigen vector -v = v sign doesnt matter

#============================================================================

#Sort Eigen values and eigen vectors in descending order
idx = np.argsort(eigen_values)[::-1] #idx = index positions sorted from biggest to smallest eigenvalue
eigen_values = eigen_values[idx] #This reorders the eigenvalues from largest to smallest using the index array.
eigen_vectors = eigen_vectors[:,idx] #Eigenvectors are columns, so we must reorder columns, take all rows.
print("Sorted Eigenvalues:\n", eigen_values, "\n")
print("Sorted Eigenvectors (PCs as columns):\n", eigen_vectors, "\n")
#========================================================================================================================

#Project Data onto principal components
#Z = mean_centred_data * W (W = matrix of eigenvectors)
Z_2D = mean_centered_data.dot(eigen_vectors[:,:2]) #first 2 eigen vectors columns
print("Projected data (PC scores):\n", Z_2D, "\n")

plt.scatter(Z_2D[:,0],Z_2D[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("2D PCA Projection")
plt.grid(True)
plt.show()

