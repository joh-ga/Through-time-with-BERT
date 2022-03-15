import faiss
import numpy as np
import umap.umap_ as umap

#Insert metadata and tensors with or without Adverbs
#lexikon = open("./data/metadataAdverb.tsv", "r")
lexikon = open("./data/metadata.tsv", "r")
tokenlist = lexikon.readlines() 
tokenlist = [s.strip() for s in tokenlist]

#data=np.genfromtxt('./data/tensorsAdverb.tsv', delimiter='\t', invalid_raise = False)
data=np.genfromtxt('./data/tensors.tsv', delimiter='\t', invalid_raise = False)
vecs  = data.astype('float32')

#dim = dimension of vector, num = amount of vectors
dim = 768                           
num =  1368                    

vecs.shape

#ncentroids = amount of clusters/centroids, niter = cluster iterations
ncentroids = 13
niter = 500

#kmeans clustering
kmeans = faiss.Kmeans(dim, ncentroids, niter=niter, verbose=True)
kmeans

kmeans.train(vecs)

# determine centroids
cent = kmeans.centroids

# get labels/indices of clusters/centroids for every vectors
labels = kmeans.index.search(vecs, 1)[1].reshape(-1).flatten()

# find index of every centroid in input (tensors.tsv)
index = faiss.IndexFlatL2(dim)
index.add(vecs)
Ic = index.search(kmeans.centroids, 1)[1].reshape(-1).flatten()
#Ic = Index of centroid

#Dimensionality reduction (PCA) for plotting
pca = faiss.PCAMatrix(dim, 2)
pca.train(kmeans.centroids)
assert pca.is_trained
centroids_2d = pca.apply_py(kmeans.centroids)
vectors_2d = pca.apply_py(vecs)

# calculate Euclidian Distances from vecs to centroids
distances = []
for i in range(num):
    distances.append(np.linalg.norm(vecs[i]-cent[labels[i]]))

# calculate Euclidian Distances between centroids and write them into csv
f = open("./data/clusterdistances"+str(ncentroids)+"c.tsv","w")
for j in range(ncentroids):
    f.write("\t" + str(j))
f.write("\n")
for i in range(ncentroids):
    f.write(str(i))
    for j in range(ncentroids):
        f.write("\t" + str(np.linalg.norm(cent[i]-cent[j])))
    f.write("\n")
f.close()

#Write Faiss results to a table in csv file
f = open("./data/overview"+str(niter)+"_"+str(ncentroids)+"c.tsv","w")
f.write("Index\tCluster\tTLNumber\tTense\tSentence\tDistanceToCent\n")
for i in range(num):
    f.write(str(i)+"\t"+str(labels[i])+"\t"+tokenlist[i]+"\t"+str(distances[i])+"\n")
f.close()



#Plotting
import matplotlib.pyplot as plt
import seaborn as sns


#PCA Plotting
plt.gcf().set_size_inches(30, 30)

sns.set(rc = {'figure.figsize':(30,30)})
pal = sns.color_palette("bright", ncentroids)

plot = sns.scatterplot(x=vectors_2d[:,0], y=vectors_2d[:,1], hue=labels, palette=pal, s=14, legend=True)
plot = sns.scatterplot(x=centroids_2d[:,0], y=centroids_2d[:,1], s=200, color = 'black', marker = 'X', legend=True)

for i in range(ncentroids):
    token = tokenlist[Ic[i]]
    plot.text(centroids_2d[i][0]+0.1, centroids_2d[i][1]+0.1, str(i)+': '+token+'(?)', horizontalalignment='left', size='medium', color='black', weight='semibold')

plot.get_figure().savefig('data/PCA_'+str(ncentroids)+'c.png')