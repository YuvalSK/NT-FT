# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 23:11:43 2018
@author: Yuval Samoilov-Katz
"""
# Cluster Analysis of FT correlation with NT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering, KMeans
        
snaps = [1,2,10,19]
clusters = 2
for snap in snaps:
    dataset = pd.read_csv('Data/snap_{}_dois_100.csv'.format(snap))
    X = dataset.iloc[:, 1:20].values
    DOIs = dataset['DOIs'].values
    base = np.arange(0,len(DOIs),1)
    
    fig, axes = plt.subplots(figsize=(16,12), nrows=1, ncols=1)    
    dend1 = shc.dendrogram(shc.linkage(X, method='ward'),labels=DOIs , orientation='right',ax=axes)
    axes.title.set_text(f'Hierarchical Clustering of Free Text and Network Topology Correlation\nSnap:{snap}')
    plt.savefig(f'Results/hei_{snap}')
    cluster = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean',linkage='ward')
    cluster.fit_predict(X)
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(X)
    
    fig, ax = plt.subplots(figsize=(14,10), nrows=3, ncols=1)    
    for i,feature in enumerate(dataset.columns[:-1]):
        ax[0].scatter(base, X[:,i], label=f'{feature}')
        ax[1].scatter(base, X[:,i], c=cluster.labels_, cmap='rainbow')
        ax[2].scatter(base, X[:,i], c=kmeans.labels_, cmap='rainbow')
    
    ax[0].title.set_text('Correlation of Free Text with Network Topology')
    ax[1].title.set_text('Hierarchical Clustering')
    ax[2].title.set_text('KMeans Clustering')
    plt.tight_layout()
    plt.savefig(f'Results/clustering_{snap}')
#ax[0].legend(loc = 'upper right')
#asso_dois = pd.read_excel('doi_to_big5.xlsx')