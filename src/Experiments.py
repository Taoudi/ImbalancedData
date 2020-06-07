from DataLoader import DataLoader
from NetworkModels import LeNet
import numpy as np
data = DataLoader()
print(data.Y)
print(data.Y.shape)
model = LeNet(data.X)
model.fit(data.X,data.Y)
loss, accuracy, precision, recall = model.predict(data.X,data.Y)

print("Recall: " + str(recall))
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Loss: " + str(loss))