import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def list_of_vertical_vectors_to_matrix(list_of_vectors, n_dims):
  matrix = np.empty([n_dims, 0])
  for vector in list_of_vectors:
    matrix = np.hstack([matrix, vector])
  return matrix

class HistogramPlot:

  def __init__(self, title, x_label, y_label):
    self.title   = title
    self.x_label = x_label
    self.y_label = y_label
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

  def add_data_series(self, array_of_values, color):
    plt.hist(array_of_values, 100, normed=1, facecolor=color, alpha=0.5)

  def show(self):
    plt.show()
