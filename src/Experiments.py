from DataLoader import DataLoader
from NetworkModels import LeNet
import numpy as np
from tensorflow.keras import metrics
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt



def ROC(precisions,recalls):
    plt.plot(precisions,recalls,label="LeNet300", marker='x') 
    plt.axis([0,1,0,1]) 
    plt.title("ROC-curve")
    plt.xlabel('Precision') 
    plt.ylabel('Recall') 
    plt.legend()
    plt.show()    

def experiment():
    METRICS = [
        metrics.TruePositives(name='tp'),
        metrics.FalsePositives(name='fp'),
        metrics.TrueNegatives(name='tn'),
        metrics.FalseNegatives(name='fn'), 
        metrics.BinaryAccuracy(name='accuracy'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc')
    ]
    data = DataLoader()
    model = LeNet(data.X,METRICS)
    model.fit(data.X,data.Y)
    RES,fpr,tpr = model.predict(data.testX,data.testY)
    model_summary(RES)
    data.summarize(True)
    ROC(fpr,tpr)
  

def model_summary(RES):
    print("TP: " + str(RES[1]))
    print("FP: " + str(RES[2]))
    print("TN: " + str(RES[3]))
    print("FN: " + str(RES[4]))

    print("Recall: " + str(RES[7]))
    print("Accuracy: " + str(RES[5]))
    print("Precision: " + str(RES[6]))
    print("AUC: " + str(RES[8]))
    print("Loss: " + str(RES[0]))

if __name__ == '__main__':
    experiment()