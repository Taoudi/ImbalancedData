from DataLoader import DataLoader
from NetworkModels import LeNet
from Augmentation import Augmenter
import numpy as np
from tensorflow.keras import metrics
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
from settings import AugmentationType


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
        plt.plot(history.history['val_precision'], label='Precision (val data)', marker='x',color='r')
        plt.plot(history.history['val_accuracy'], label='Accuracy (val data)',marker='+',color='r')
        plt.plot(history.history['val_recall'], label='Recall (val data)',marker='.',color='r')

        plt.plot(history.history['precision'], label='Precision (training data)', marker='x',color='g')
        plt.plot(history.history['accuracy'], label='Accuracy (training data)',marker='+',color='g')
        plt.plot(history.history['recall'], label='Recall (training data)',marker='.',color='g')

        plt.title('Binary Cross Entropy for Credit Card Fraud\n ' + self.augmentation.type_text)
        plt.ylabel('Percentage')
        plt.xlabel('No. Epochs')
        plt.legend(loc="upper left")
        plt.show()

    def experiment(self,under=False,ratio=3):
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
        augmenter = Augmenter(data.X,data.Y)

        if under:
            data.X,data.Y = augmenter.undersample(ratio=ratio)

        if self.augmentation.type == 1 or self.augmentation.type == 2:
            data.X, data.Y = augmenter.duplicate(noise=self.augmentation.noise,sigma=self.augmentation.sigma)
        elif self.augmentation.type == 3:
            data.X, data.Y = augmenter.SMOTE()


        #data.normalize()
        #print(len(data.X))
        #print(len(data.valX))

        data.summarize(test=False)
        his = model.fit(data.X,data.Y,data.valX, data.valY)
        RES,fpr,tpr = model.predict(data.testX,data.testY)
        #self.model_summary(RES)
        #self.plot(his)
        #self.ROC(fpr,tpr)
        return RES[8]

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


    def undersample_experiment(self):
        jumps = 10
        iters = 5*jumps
        ratios = []
        AUCs = []
        for i,j in enumerate(range(1,iters+1,10)):
            #ratio = j*5-4
            #ratio = 2**j
            res = experiment.experiment(under=True,ratio=ratio)
            ratios.append(ratio)
            AUCs.append(res)
            print(i , j )
        print(ratios, AUCs)
        plt.scatter(ratios,AUCs)
        plt.title('AUC for increasing data unbalance (using undersampling)\n')
        plt.ylabel('Area under the curve (AUC)')
        plt.xlabel('Ratio between majority and minority classes')
        plt.show()

        #print(ratios)
        #print(AUCs)

if __name__ == '__main__':
   experiment = Experiment(0,sigma=0.00)
   experiment.undersample_experiment()