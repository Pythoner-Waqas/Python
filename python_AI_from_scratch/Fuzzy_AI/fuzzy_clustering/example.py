import pandas as pd
import matplotlib.pyplot as plt

from FUZZY import CLUSTERING

train = pd.read_excel('walkertrain.xlsx')
test = pd.read_excel('walkertest.xlsx')

data_train = train[['x','y','v']]
data_test = test[['x','y','v']]
X_train = data_train.values

X = train[['x','y','v']].values
plt.scatter(X[:,0], X[:,1], c=X[:,2],cmap="Reds")

Fclustering = CLUSTERING(method="Gustafson–Kessel")
memberships, classCenters = Fclustering.fit(X)
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].scatter(X[:,0], X[:,1], c=memberships[:,0],cmap="Reds")
axs[1].scatter(X[:,0], X[:,1], c=memberships[:,1],cmap="Reds")


Fclustering = CLUSTERING(method="Cmeans",n_clusters = 3)
memberships, classCenters = Fclustering.fit(X)
fig, axs = plt.subplots(3)
fig.suptitle('Vertically stacked subplots')
axs[0].scatter(X[:,0], X[:,1], c=memberships[:,0],cmap="Reds")
axs[1].scatter(X[:,0], X[:,1], c=memberships[:,1],cmap="Reds")
axs[2].scatter(X[:,0], X[:,1], c=memberships[:,2],cmap="Reds")
