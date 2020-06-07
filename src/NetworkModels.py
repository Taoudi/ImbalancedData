
import tensorflow as tf
from tensorflow.keras import datasets,layers,models,initializers
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class LeNet:
    def __init__(self,X):
        self.model = models.Sequential()
        self.model.add(layers.Flatten(input_shape=(X[0].shape)))
        self.model.add(layers.Dense(300,activation='relu'))
        self.model.add(layers.Dense(100,activation='relu'))
        self.model.add(layers.Dense(10,activation='relu'))
        self.model.compile(tf.keras.optimizers.Adam(learning_rate=2e-4),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    def fit(self,X,Y):
        history = self.model.fit(X, Y, epochs=10, batch_size=64,validation_split=0.2)
        return history
    
    def predict(self, X, Y):
        test_loss, test_acc = self.model.evaluate(X, Y, verbose=2)
        predictions = self.model.predict(X)
        print(predictions.shape)
        TP = tf.math.count_nonzero(predictions * Y)
        TN = tf.math.count_nonzero((predictions - 1) * (Y - 1))
        FP = tf.math.count_nonzero(predictions * (Y - 1))
        FN = tf.math.count_nonzero((predictions - 1) * Y)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return test_loss, test_acc, precision, recall