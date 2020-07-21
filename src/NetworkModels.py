
import tensorflow as tf
from tensorflow.keras import datasets,layers,models,initializers,metrics,losses
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve,roc_auc_score

class LeNet:
 
    def __init__(self,X, metrics):
        self.model = models.Sequential()
        self.model.add(layers.Flatten(input_shape=(X[0].shape)))
        self.model.add(layers.Dense(300,activation='relu'))
        self.model.add(layers.Dense(100,activation='relu'))
        self.model.add(layers.Dense(1,activation='sigmoid'))
        self.model.compile(tf.keras.optimizers.Adam(learning_rate=2e-4),loss=losses.BinaryCrossentropy(),
              metrics=metrics)

    def fit(self,X,Y,valX,valY):
        history = self.model.fit(X, Y, epochs=20, batch_size=128,validation_data=(valX,valY),verbose=1)
        return history
    
    def predict(self, X, Y):
        RESULTS = self.model.evaluate(X, Y, verbose=2)
        print(RESULTS)
        predictions = self.model.predict(X)
        fpr,tpr,thresholds = roc_curve(Y , predictions)
        """predictions = tf.argmax(predictions, 1)
        TP = tf.math.count_nonzero(predictions * Y)
        TN = tf.math.count_nonzero((predictions - 1) * (Y - 1))
        FP = tf.math.count_nonzero(predictions * (Y - 1))
        FN = tf.math.count_nonzero((predictions - 1) * Y)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)"""
        return RESULTS, fpr,tpr