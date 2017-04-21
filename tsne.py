
# coding: utf-8

# In[1]:

import torch
import numpy as np
import model
import data
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:

corpus_2 = data.Corpus('./data/gutenberg',30000)
corpus_1 = data.Corpus('./data/penn',10000)


# In[4]:

row_1=[]
row_1.append(corpus_1.dictionary.word2idx['queen'])
row_1.append(corpus_1.dictionary.word2idx['female'])
row_1.append(corpus_1.dictionary.word2idx['male'])
row_1.append(corpus_1.dictionary.word2idx['king'])
row_1.append(corpus_1.dictionary.word2idx['man'])
row_1.append(corpus_1.dictionary.word2idx['women'])
row_1.append(corpus_1.dictionary.word2idx['apple'])
row_1.append(corpus_1.dictionary.word2idx['orange'])
row_1


# In[5]:

row_2=[]
row_2.append(corpus_2.dictionary.word2idx['queen'])
row_2.append(corpus_2.dictionary.word2idx['female'])
row_2.append(corpus_2.dictionary.word2idx['male'])
row_2.append(corpus_2.dictionary.word2idx['king'])
row_2.append(corpus_2.dictionary.word2idx['man'])
row_2.append(corpus_2.dictionary.word2idx['women'])
row_2.append(corpus_2.dictionary.word2idx['apple'])
row_2.append(corpus_2.dictionary.word2idx['orange'])
row_2


# In[8]:

import pickle
with open('penn_lstm.pk')as f:
    Embeddings_to_plot = pickle.load(f)


# In[9]:

from sklearn.manifold import TSNE
TSNE_model = TSNE(n_components=2,random_state=0)
represent2D = TSNE_model.fit_transform(Embeddings_to_plot)


# In[60]:
plt.figure(figsize=(15,15))
plt.scatter(represent2D[:,0],represent2D:,1],marker="x", s=0.8,color='blue')
plt.title("TSNE_plot_penn")
plt.savefig("TSNE_plot_penn.png")
plt.show()


# In[58]:

import pylab
from sklearn.cluster import KMeans
kmeans = KMeans(init='k-means++', n_clusters=30, n_init=20)
kmeans.fit(represent2D)


# In[67]:

plt.figure(figsize=(25,25))
pylab.scatter(represent2D[:,0],represent2D:,1],s=6,c=kmeans.labels_,cmap=pylab.cm.Accent)
pylab.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='r')
plt.scatter(represent2D[row_1,0],represent2D[row_1,1],marker="x", s=50,color='blue')
for i in range(8):
    plt.annotate(word[i],(represent2D[row_1[i],0],rrepresent2D[row_1[i],1]))
plt.title("T-SNE with Kmean Clustering")
plt.savefig("TSNE_Kmean_plot_Penn.png")
plt.show()


# In[26]:

import pickle
with open('gutenberg_lstm.pk')as f:
    Embeddings_2 = pickle.load(f)


# In[27]:

from sklearn.manifold import TSNE
TSNE_model = TSNE(n_components=2,random_state=0)
repre2D = TSNE_model.fit_transform(Embeddings_2)


# In[45]:

plt.figure(figsize=(15,15))
plt.scatter(repre2D[:,0],repre2D[:,1], s=0.5,color='blue')
plt.title("TSNE_plot_Gutenberg")
plt.savefig("TSNE_plot_Gutenberg.png")
plt.show()


# In[29]:

import pylab
from sklearn.cluster import KMeans
kmeans_2 = KMeans(init='k-means++', n_clusters=30, n_init=10)
kmeans_2.fit(repre2D)


# In[41]:

plt.figure(figsize=(25,25))
pylab.scatter(repre2D[:,0],repre2D[:,1],s=2,c=kmeans_2.labels_,cmap=pylab.cm.Accent)
pylab.scatter(kmeans_2.cluster_centers_[:,0],kmeans_2.cluster_centers_[:,1],c='r')
plt.scatter(repre2D[row_2,0],repre2D[row_2,1],marker="x", s=50,color='blue')
for i in range(8):
    plt.annotate(word[i],(repre2D[row_2[i],0],repre2D[row_2[i],1]))
plt.title("T-SNE with Kmean Clustering")
plt.savefig("TSNE_Kmean_plot_Gutenberg.png")
plt.show()


# In[ ]:



