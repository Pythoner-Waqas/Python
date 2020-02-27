
import pandas as pd
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler

# Real data
filename = 'cars_clus.csv'
pdf = pd.read_csv(filename)

#data cleaning
print ("Shape of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", pdf.size)
pdf.head(5)

featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
x = featureset.values #returns a numpy array

min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)

dist_matrix = distance_matrix(feature_mtx,feature_mtx)

Z = hierarchy.linkage(dist_matrix, 'complete')

import pylab
def leafLabelFunc(index):
    return '[%s %s %s]' % (pdf['manufact'][index], pdf['model'][index], int(float(pdf['type'][index])) )
fig = pylab.figure(figsize=(18,50))
dendro = hierarchy.dendrogram(Z,  leaf_label_func=leafLabelFunc, leaf_rotation=0, leaf_font_size =12, orientation = 'right')

