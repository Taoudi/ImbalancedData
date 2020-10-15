from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from DataLoader import DataLoader
import math
class Augmenter:

    def get_counts(self):
        # Calculates the amounts of instances of minority and majority in the main dataset, along with how many augmented samples should be added
        unique, counts = np.unique(self.Y, return_counts=True)
        arg_min_count = int(np.argmin(counts))
        arg_max_count = int(np.argmax(counts))
        min_count = int(np.min(counts))
        max_count = int(np.max(counts))
        n_augmented = int(np.abs(min_count-max_count))
        return n_augmented,min_count,max_count

    def get_indices(self,max_count,min_count):
        # Returns a list of indices for the minority classes in the main dataset
        return np.argpartition(-self.Y,min_count)[:min_count]

    def update_data(self,X,Y):
        self.X = X
        self.Y = Y
        n_augmented, min_count,max_count = self.get_counts()
        arg_partition = self.get_indices(max_count,min_count)

        self.minority_data = self.X[arg_partition]
        self.minority_dataY = self.Y[arg_partition]

    def __init__(self,X,Y):
        self.update_data(X,Y)

    def set_data(self,X,Y):
        self.X = X
        self.Y = Y
        n_augmented, min_count,max_count = self.get_counts()
        arg_partition = self.get_indices(max_count,min_count)
        self.minority_data = self.X[arg_partition]

    def undersample(self,ratio=100):
        n_augmented, min_count,max_count = self.get_counts()
        augmented = np.zeros((n_augmented,self.X.shape[1]))
        
        p = np.argpartition(self.Y,-max_count)[:max_count]
        np.random.shuffle(p)
        chosen = np.random.choice(a=p,size=min_count*ratio,replace=False).astype(int)

        newY = np.concatenate((self.Y[chosen], self.minority_dataY))
        newX = np.concatenate((self.X[chosen], self.minority_data))
        self.update_data(newX,newY)

        return newX, newY

    def duplicate(self ,noise=False, sigma=0.01, mu=0):
        # Oversampling through duplication, minority classes are randomly chosen and duplicated until dataset is balanced
        n_augmented, min_count,max_count = self.get_counts()
        augmented = np.zeros((n_augmented,self.X.shape[1]))
        augmentedY = np.zeros(n_augmented)
        arg_partition = self.get_indices(max_count,min_count)
        for i in range(n_augmented):
            idx = np.random.randint(1,min_count)
            augmented[i] = self.X[arg_partition[idx]]
            augmentedY[i] = self.Y[arg_partition[idx]]
            if noise:
                augmented[i]+= np.random.normal(mu, sigma, self.X[0].shape)
        newX = np.concatenate((self.X, augmented))
        newY = np.concatenate((self.Y, augmentedY))
        unique, counts = np.unique(newY, return_counts=True)
        self.update_data(newX,newY)
        return newX, newY

    # Returns indices of k-nearest neighbours in the self.minority_data matrix for each minority point
    def k_nearest(self,k):
            neighbours = np.zeros((self.minority_data.shape[0],k+1))
            neighbours2 = np.zeros((self.minority_data.shape[0],k))
            k_distances = np.zeros((self.minority_data.shape[0],k))
            # n_attributes = len(point)
            created_samples = 0
            for i in range(self.minority_data.shape[0]):
                distances = np.linalg.norm(self.minority_data[i] - self.minority_data,axis=1)
                neighbours[i] = np.argpartition(distances,k+1)[:k+1]
                neighbours2[i] = neighbours[i][neighbours[i]!=i]
                k_distances[i] = distances[neighbours2[i].astype(int)]

            return neighbours2,k_distances

    def SMOTE(self):
        # Oversampling through generating samples along neighbours
        n_augmented, min_count,max_count = self.get_counts()
        N = int(np.ceil(len(self.X) - len(self.minority_data))/len(self.minority_data))
        # N = int(np.floor(N/100)) # Integral multiples of 100

        if(N>10):
            k = int(N*np.log(N))
        else:
            k = int(5*N)

        if N < 1:
            print("N must be atleast 100")

        print(N)
        print(k)
        print(N*len(self.minority_data))
        arg_neighbours, k_distances = self.k_nearest(k)
        scalars = np.random.rand(arg_neighbours.shape[0], N)
        samples = np.zeros((arg_neighbours.shape[0], N, self.minority_data.shape[1]))
        rng = np.random.choice(k,size=N,replace=False).astype(int)


        for i in range(len(self.minority_data)):
            rng = np.random.choice(k,size=N,replace=False).astype(int)
            samples[i] = self.minority_data[i][:] + np.expand_dims(scalars[i],axis=1) * (self.minority_data[i][:] -self.minority_data[rng][:])
        
        samples = samples.reshape((samples.shape[0]*samples.shape[1], samples.shape[2]))
        print(samples.shape)

        samplesY = np.ones(len(samples))
        newX = np.concatenate((self.X, samples))
        newY = np.concatenate((self.Y, samplesY))
        self.update_data(newX,newY)
        return newX, newY


if __name__ == '__main__':
    data = DataLoader()
    aug = Augmenter(data.X, data.Y)

    aug.undersample()