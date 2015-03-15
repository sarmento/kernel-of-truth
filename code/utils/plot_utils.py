import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def list_of_vertical_vectors_to_matrix(list_of_vectors, n_dims):
  matrix = np.empty([n_dims, 0])
  for vector in list_of_vectors:
    matrix = np.hstack([matrix, vector])
  return matrix

def plot_histogram(array_of_values, title, color, x_label, y_label, is_to_show, output_file):
  plt.hist(array_of_values, 100, normed=0, facecolor=color, alpha=0.5)
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  if is_to_show:
    plt.show()


