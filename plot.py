import matplotlib as mpl
#mpl.use('Agg')
import numpy as np
import pylab
import reductions
from matplotlib.colors import ListedColormap
from datasets import Datasets
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from constants import *
import os

dts = Datasets(0)

def plot_data(filename, label, dataset, dimension=2, reduction='PCA'):
    data = dts.load_dataset(dataset, train_size=0)
    (X, Y) = (data['x_test'], data['y_test'])

    X = scale(X)
    # Reduce the dimensionality to the given value
    red_func = getattr(reductions, reduction)
    X = red_func(X, dimension)

    if dimension == 2:
        plot_2d(filename, label, X, Y)

def plot_2d(filename, label, X, Y, Z=None):
    assert X.shape[0] > 2, "Only two dimensional data allowed. Apply reduction to data first"
    h = 0.02
    figure = pylab.figure(figsize=(15, 10))
    cmap = pylab.cm.winter
    cmap.set_under("magenta")
    cmap.set_over("yellow")

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                  np.arange(y_min, y_max, h))

    pl = pylab.subplot(1, 1, 1)
    # Plot the training points
    pl.scatter(X[:, 0], X[:, 1], c=Y, s=60, cmap=cmap)
    pl.set_xlim(xx.min(), xx.max())
    pl.set_ylim(yy.min(), yy.max())
    pl.set_xticks(np.linspace(x_min, x_max, 10))
    pl.set_yticks(np.linspace(y_min, y_max, 10))
    pl.set_title(label)
    figure.savefig(filename)

def plot_confusion_matrix(filename, y_test, y_pred, label):
    cm = confusion_matrix(y_test, y_pred)

    pylab.subplot(111)
    pylab.matshow(cm)
    # Show confusion matrix in a separate window
    pylab.title(label, fontsize=20)
    pylab.colorbar()
    pylab.ylabel('True label', fontsize=20)
    pylab.xlabel('Predicted label', fontsize=20)
    pylab.savefig(filename)

def plot_metric(filename, type, y_test, y_pred, dataset, algorithm, training_size):
    label = "%s-%s-size-%d" % (dataset, algorithm, training_size)
    plot_confusion_matrix(filename, y_test, y_pred, label)


def plot_cv(algorithm, dataset, training_size, cv_results, cv_dir):
    label = "cross_validation-%s-%s-size-%s" % (algorithm, dataset, training_size)
    filename = os.path.join(cv_dir, "cv-%s-%s-size-%s.png" % (algorithm, dataset, training_size))
    return
    n_dim = len(cv_results[0][0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
        xs = np.arange(20)
        ys = np.random.rand(20)

        # You can provide either a single color or an array. To demonstrate this,
        # the first bar of each set will be colored cyan.
        cs = [c] * len(xs)
        cs[0] = 'c'
        ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    pylab.show()


def plot_training_results(filename, train_sizes, scores):
    width = 7
    fig = pylab.figure()
    ax = pylab.subplot(111)
    ax.bar(train_sizes, scores, width, color='b')
    ax.set_ylabel('Scores')
    ax.set_xticks(train_sizes)
    ax.set_title('Accuracy vs Training Size')
    fig.show()
    fig.savefig(filename)

if __name__ == "__main__":
    plot_data("/tmp/test_plot", "Label:Iris", "iris")
