from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class Augmenter:
    def __init__(self,X):
        self.X = X
    
    def duplicate(self, X, Y):
        unique, counts = np.unique(Y, return_counts=True)
        arg_min_count = int(np.argmin(counts))
        arg_max_count = int(np.argmax(counts))
        min_count = int(np.min(counts))
        max_count = int(np.max(counts))

        n_augmented = int(np.abs(min_count-max_count))

        augmented = np.zeros((n_augmented,X.shape[1]))
        augmentedY = np.zeros(n_augmented)
        arg_partition = np.argpartition(-Y,min_count)[:min_count]
        partition = -np.partition(-Y,min_count)[:min_count]
        for i in range(n_augmented):
            idx = np.random.randint(1,min_count)
            augmented[i] = X[arg_partition[idx]]
            augmentedY[i] = Y[arg_partition[idx]]

        newX = np.concatenate((X, augmented))
        newY = np.concatenate((Y, augmentedY))
        unique, counts = np.unique(newY, return_counts=True)



        return newX, newY
