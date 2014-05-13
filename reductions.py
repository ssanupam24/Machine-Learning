from sklearn import decomposition
from sklearn import manifold

def PCA(X, dimensions):
    return decomposition.PCA(n_components=dimensions ).fit_transform(X)

def LinearEmbedding(X, dimension):
    n = int(X.shape[0] * 0.01)
    return manifold.Isomap(20, dimension).fit_transform(X)


