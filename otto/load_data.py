__author__ = 'spszabad'
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import numpy as np


train_file = 'data/train.csv'

data = pd.read_csv(train_file)

X = data.as_matrix(list(data.columns.values[1:-1]))

kmeans = MiniBatchKMeans(init='k-means++', n_clusters=100, n_init=10)

kmeans.fit(X)

y = kmeans.transform(X)
d = np.zeros(y.shape)

for i in range(y.shape[0]):
    v = y[i,:]
    delta = np.std(v)/100
    v = np.exp(-v/delta)
    v_sum = np.exp(-v/delta).sum()
    d[i,:] = v/v_sum

print 'Done'






