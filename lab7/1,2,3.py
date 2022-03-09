import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

#question1
print("Q1:")
#reading to daatframe
df = pd.read_csv("Iris.csv")
# Extracting features
features = list(df)
features.remove("Species")
x = df.loc[:, features].values

#applying pca
#specifiying number of components = 2
pca = PCA(n_components=2)
#fititng and trsasforming
principalComponents = pca.fit_transform(x)
#converting to dataframe
pca_df = pd.DataFrame(data = principalComponents, columns = ['component1', 'component2'])
#another pca df which also hs target values
pca_df_with_target = pd.concat([pca_df, df['Species']], axis = 1)

#to get eignevalues and eigen matrix
cov_matrix = np.cov(x.T)
# Calculate Eigenvalues and Eigenmatrix
eigenvalues, eigenvectors = eig(cov_matrix)
print("eigenvectors" , eigenvectors , "\nEigenvalues" ,  eigenvalues)
#
#plottig eigenvalues v/s components graph
n = np.linspace(1, 4, 4)
plt.plot(n , eigenvalues)
plt.xticks(n , n)
plt.ylabel("Eigenvalues")
plt.xlabel("Components")
plt.title("Eigenvalues plot")
plt.show()

# #plottig 2-d data
col_names = list(pca_df)
plt.scatter(pca_df[col_names[0]] , pca_df[col_names[1]])
plt.ylabel("Column2")
plt.xlabel("Column1")
plt.title("2-d Plot of reduced dimention data")
plt.show()

#question2
print("\nQ2:")

#k means clusterin
from sklearn.cluster import KMeans
#created copt of recued data
data = pca_df.copy()
K = 3
#fitting data accordignt o number of clusters given
kmeans = KMeans(n_clusters=K)
kmeans.fit(data)
#making precition of labels of data points
kmeans_prediction = kmeans.predict(data)

#a part
#making precition of labels of data points
data["Labels"] = kmeans_prediction
d1 = data[data["Labels"] == 1]
d0 = data[data["Labels"] == 0]
d2 = data[data["Labels"] == 2]
col_names = list(data)
#to get centre point of data
centroids = kmeans.cluster_centers_
plt.scatter(d1[col_names[0]] , d1[col_names[1]] , label = "Iris-setosa")
plt.scatter(d2[col_names[0]] , d2[col_names[1]] , label = "Iris-versicolor")
plt.scatter(d0[col_names[0]] , d0[col_names[1]] , label = "Iris-virginica")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("clustering plot")
#plottign the data along with their center points
plt.scatter(centroids[:,0] , centroids[:,1] , s = 70, color = "k" , marker = "o" , label = "Center")
plt.legend()
plt.show()

#b part
print("Q2: b part")
#fidning distortion measure
print(round(kmeans.inertia_ , 3))

print("\nC Part")
#to find prity score
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # print(contingency_matrix)

    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)

print("purity score for k = 3 is" , purity_score(df["Species"] ,kmeans_prediction).round(3))

print("\nQ3 begins")
#Q3
print("Distortion measure")
K = np.arange(2 , 8 , 1)

dist_list = []
purity_scores= []
for k in K :
    #copying pca reduced data
    data = pca_df.copy()
    # fitting data accordignt o number of clusters given
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    # making precition of labels of data points
    kmeans_prediction = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    #to get purity score
    purity_scores.append(purity_score(df["Species"] , kmeans_prediction).round(3))
    dist = round(kmeans.inertia_ , 3)
    dist_list.append(dist)

print(((dist_list )))

#outputting the k value and purity score as dataframe
k_dict = {"k_value" : K , "Purity Scores" : purity_scores}
k_df = pd.DataFrame.from_dict(k_dict )
print("\n" , k_df)

#plotting k value v/s purity
plt.plot(K , purity_scores)
plt.xlabel("K value")
plt.ylabel("Purity Scores")
plt.title("K value v/s Purity Scores")
plt.show()

#plotting k value v/s no. of clusters
plt.plot(K , dist_list)
plt.xlabel("No. of clsuters")
plt.ylabel("Distortion Measure")
plt.title("No. of clsuters v/s Distortion Measure")
plt.show()
