from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class Augmenter:
    def __init__(self,X):
        self.X = X

    def get_indices(self,Y,max_count,min_count):
        # Returns a list of indices for the minority classes in the main dataset
        return np.argpartition(-Y,min_count)[:min_count]

    def get_counts(self,Y):
        # Calculates the amounts of instances of minority and majority in the main dataset, along with how many augmented samples should be added
        unique, counts = np.unique(Y, return_counts=True)
        arg_min_count = int(np.argmin(counts))
        arg_max_count = int(np.argmax(counts))
        min_count = int(np.min(counts))
        max_count = int(np.max(counts))
        n_augmented = int(np.abs(min_count-max_count))
        return n_augmented,min_count,max_count
    def duplicate(self, X, Y):
        # Oversampling through duplication, minority classes are randomly chosen and duplicated until dataset is balanced
        n_augmented, min_count,max_count = self.get_counts(Y)
        augmented = np.zeros((n_augmented,X.shape[1]))
        augmentedY = np.zeros(n_augmented)
        arg_partition = self.get_indices(Y,max_count,min_count)
        for i in range(n_augmented):
            idx = np.random.randint(1,min_count)
            augmented[i] = X[arg_partition[idx]]
            augmentedY[i] = Y[arg_partition[idx]]

        newX = np.concatenate((X, augmented))
        newY = np.concatenate((Y, augmentedY))
        unique, counts = np.unique(newY, return_counts=True)



        return newX, newY
