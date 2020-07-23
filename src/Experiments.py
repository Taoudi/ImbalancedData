from DataLoader import DataLoader
from NetworkModels import LeNet
from Augmentation import Augmenter
import numpy as np
from tensorflow.keras import metrics
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
from settings import AugmentationType
import autokeras as ak
from sklearn.metrics import roc_curve,roc_auc_score


class Experiment:

    def __init__(self, aug_type, sigma=0.01):
        self.augmentation = AugmentationType(aug_type,sigma=sigma)

    def ROC(self,precisions,recalls):
        plt.plot(precisions,recalls,label="LeNet300", marker='x') 
        plt.axis([0,1,0,1]) 
        plt.title("ROC-curve\n " + self.augmentation.type_text)
        plt.xlabel('Precision') 
        plt.ylabel('Recall') 
        plt.legend()
        plt.show()

    def plot(self,history):
        plt.plot(history.history['val_precision'], label='Precision (val data)', marker='x')
        plt.plot(history.history['val_accuracy'], label='Accuracy (val data)',marker='+')
        plt.plot(history.history['val_recall'], label='Recall (val data)',marker='.')
        plt.title('Binary Cross Entropy for Credit Card Fraud\n ' + self.augmentation.type_text)
        plt.ylabel('Percentage')
        plt.xlabel('No. Epochs')
        plt.legend(loc="upper left")
        plt.show()

    def experiment(self):
        METRICS = [
            metrics.TruePositives(name='tp'),
            metrics.FalsePositives(name='fp'),
            metrics.TrueNegatives(name='tn'),
            metrics.FalseNegatives(name='fn'), 
            metrics.BinaryAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
            metrics.AUC(name='AUC')
        ]

        data = DataLoader()
        #model = LeNet(data.X,METRICS)
        augmenter = Augmenter(data.X)
        model = ak.StructuredDataClassifier(overwrite=True,max_trials=10,metrics=METRICS,objective="val_AUC")
        if self.augmentation.type == 1 or self.augmentation.type == 2:
            data.X, data.Y = augmenter.duplicate(data.X,data.Y,noise=self.augmentation.noise,sigma=self.augmentation.sigma)
        elif self.augmentation.type == 3:
            data.X, data.Y = augmenter.SMOTE()



        #his = model.fit(data.X,data.Y,data.valX, data.valY)
        #RES,fpr,tpr = model.predict(data.testX,data.testY)
        model.fit(data.X,data.Y,validation_data=(data.valX, data.valY),epochs=20)
        his = model.final_fit(data.X,data.Y,validation_data=(data.valX, data.valY), retrain=True)
        print("HELLO")
        predictions = model.predict(data.testX)
        RES = model.evaluate(data.testX, data.testY)
        fpr,tpr,thresholds = roc_curve(data.testY , predictions)

        #self.model_summary(RES)
        #data.summarize(True)
        self.plot(his)
        self.ROC(fpr,tpr)
    

    def model_summary(self,RES):
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
    experiment = Experiment(2)
    experiment.experiment()