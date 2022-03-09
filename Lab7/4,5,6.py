from sklearn.decomposition import PCA
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import pandas as pd
import scipy.stats
import numpy as np
from sklearn.cluster import DBSCAN

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

print("Q4")
#q4
from sklearn.mixture import GaussianMixture
data = pca_df.copy()

# a part
#creatig gmm model with 3 cluster
gmm = GaussianMixture(n_components=3)
#fititng data
gmm.fit(data)
#predicitng labels of data
GMM_prediction = gmm.predict(data)
data["Labels"] = GMM_prediction

X = data.copy()

#finding cnetres of clusters
C = np.array([[0.8, -0.1], [0.2, 0.4]])
X = np.array(X.drop("Labels" , axis = 1))
#creating empty centers array to store centre values
centers = np.empty(shape=(gmm.n_components, X.shape[1]))
for i in range(gmm.n_components):
    density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(X)
    centers[i, :] = X[np.argmax(density)]
#pritning clusters as dataframe
centers = pd.DataFrame(centers)
X = pd.DataFrame(X)

#making sctater plot of data poits with center
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=GMM_prediction, s=40, cmap='viridis' )


d1 = data[data["Labels"] == 1]
d0 = data[data["Labels"] == 0]
d2 = data[data["Labels"] == 2]
col_names = list(data)
#to get centre point of data
#
plt.scatter(d1[col_names[0]] , d1[col_names[1]] , label = "Iris-setosa")
plt.scatter(d2[col_names[0]] , d2[col_names[1]] , label = "Iris-versicolor")
plt.scatter(d0[col_names[0]] , d0[col_names[1]] , label = "Iris-virginica")
plt.scatter(centers.iloc[:, 0], centers.iloc[:, 1], s=70 , c = "k" , label = "Centre")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.title("clustering plot")
plt.show()

#b part
print("\nQ4 b")
#findin distotion measure
print((gmm.score(X)*len(data)).round(3))

#c part
print("\nQ4 c")
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # print(contingency_matrix)

    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)

print(purity_score(df["Species"] , GMM_prediction ))

print("\nQ5:")
K = np.arange(2 , 8 , 1)

purity_scores = []
distortion_measures = []


for k in K :
    #copying pca redcued data
    data = pca_df.copy()
    # creatig gmm model with n clusters
    gmm = GaussianMixture(n_components=k , random_state= 42)
    gmm.fit(data)
    #predciitng labels
    GMM_prediction = gmm.predict(data)
    #getting purity scores
    purity_scores.append(purity_score(df["Species"], GMM_prediction).round(3))
    # gmm.lower_bound_(10)
    distortion_measures.append((gmm.score(data).round(3)*len(data)))

#printing prity scores and distortion measures
k_dict = {"k_value" : K , "Purity Scores" : purity_scores , "distortion_measures":distortion_measures}
k_df = pd.DataFrame.from_dict(k_dict )
print("\n" , k_df)

#getting graph of k v/s purty scores
plt.plot(K , purity_scores)
plt.xlabel("K value")
plt.ylabel("Purity Scores")
plt.title("K value v/s Purity Scores")
plt.show()

#getting graph of k v/s distortion measures
plt.plot(K , distortion_measures)
plt.xlabel("K value")
plt.ylabel("distortion_measures")
plt.title("K value v/s distortion_measures")
plt.show()

print("\nQ6:")
#Q6 begins
def dbs(e, ms):
    #creating the dbscan model
    dbscan_model=DBSCAN(eps=e, min_samples = ms).fit(pca_df)
    DBSCAN_predictions = dbscan_model.labels_
    return (DBSCAN_predictions)

EPS = [1,5]
MS = [4 , 10]

purity= []
e = 1
#finding the purity score and plottign the scatter plot of data points
#for different values of eps and min samples
for ms in MS:
    data["Labels"] =dbs(e , ms)
    d1 = data[data["Labels"] == 1]
    d0 = data[data["Labels"] == 0]
    col_names = list(data)
    #plottign the scatter plots of data points
    plt.scatter(d1[col_names[0]], d1[col_names[1]])
    plt.scatter(d0[col_names[0]], d0[col_names[1]])
    #getting purity scores
    purity.append(purity_score(df["Species"] ,dbs(e , ms)).round(3))
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    title_ = "DBSCAN clustering plot" + "for EPS: " + str(e) + ", Min socre: " + str(ms)
    plt.title(title_)
    plt.show()

e = 5
for ms in MS:
    data["Labels"] = dbs(e, ms)
    d0 = data[data["Labels"] == 0]
    col_names = list(data)
    #plottign the scatter plots of data points
    plt.scatter(d0[col_names[0]], d0[col_names[1]])
    #getting purity scores
    purity.append(purity_score(df["Species"] ,dbs(e , ms) ).round(3))
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    title_ = "DBSCAN clustering plot" + "for EPS: " + str(e) + ", Min socre: " + str(ms)
    plt.title(title_)
    plt.show()

#printign the dataframe of purity scores
k_dict = {"E value" : [1 , 1, 5, 5] , "MS Value"  :[4, 10,4, 10] , "Purity Scores" : purity }
k_df = pd.DataFrame.from_dict(k_dict )
print("\n" , k_df)