import numpy as np
from keras.utils import Sequence
class MyBatchGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, X, y, batch_size=1, shuffle=True):
        'Initialization'
        self.X = X
        self.y = y
        # self.d = d
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y)/self.batch_size))

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        Xb = np.empty((self.batch_size, *self.X[index].shape))
        yb = np.empty((self.batch_size, *self.y[index].shape))
        # db = np.empty((self.batch_size, *d[index].shape))
        # naively use the same sample over and over again
        for s in range(0, self.batch_size):
            Xb[s] = self.X[index]
            yb[s] = self.y[index]
            # db[s] = d[index]
        return Xb, yb
  
  ## helper functions for doc2vec algorithm
def my_split(window_size):
    def _lambda(tensor):
        import tensorflow as tf
        return tf.split(tensor, window_size + 1, axis=1)
    return _lambda


def squeeze(axis=-1):
    def _lambda(tensor):
        import tensorflow as tf
        return tf.squeeze(tensor, axis=axis)
    return _lambda


def stack(window_size):
    def _lambda(tensor):
        import tensorflow as tf
        return tf.stack([tensor] * window_size, axis=1)
    return _lambda

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, roc_auc_score, roc_curve

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
  
def plot_learning_graphs(results):
  epochs = len(results['val_loss'])
  x_axis = range(0, epochs)
  # plot log loss
  fig, ax = plt.subplots()
  ax.plot(x_axis, results['loss'], label='Train')
  ax.plot(x_axis, results['val_loss'], label='Test')
  ax.legend()
  plt.ylabel('Log Loss')
  plt.title('Log Loss')
  plt.show()

  # plot classification error
  fig, ax = plt.subplots()
  ax.plot(x_axis, results['acc'], label='Train')
  ax.plot(x_axis, results['val_acc'], label='Test')
  ax.legend()
  plt.ylabel('Classification Accuracy')
  plt.title('Accuracy')
  plt.show()
