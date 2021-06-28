from matplotlib import markers
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
        plt.plot(precisions,recalls,label="LeNet300") 
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

    def experiment(self,under=False,ratio=3,plot=False):
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
        if plot:
            self.plot(his)
            self.ROC(fpr,tpr)
        return RES

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

    def duplication_experiment(self):
        iters = 8
        rounds = 3
        AUCs = np.zeros((iters, rounds))
        res = experiment.experiment(under=False,plot=True)
        print(res)

    def undersample_experiment(self):
        #jumps = 10
        iters = 8
        rounds = 3
        ratios = []
        #AUCs = []
        AUCs = np.zeros((iters, rounds))
        #ratios = np.arange(1,iters+1)
        #ratios = np.power(ratios,2)
        #AUCs = np.random.rand(iters,rounds)
        print("SHAPE", np.mean(AUCs,axis=1).shape)
       

        #plt.errorbar(ratios, np.mean(AUCs,axis=1),yerr=deviation,ecolor='g', color='g',marker='o')
       
        for r in range(rounds):
            for i,j in enumerate(range(1,iters+1)):
                #ratio = j*5-4
                #ratio = 2**j
                ratio=j**2
                #ratio = int(np.log2(j+1))
                print("RATIO" ,ratio, "ROUND", r)
                res = experiment.experiment(under=True,ratio=ratio)
                if r == 0:
                    ratios.append(ratio)
                AUCs[i][r] = res[8]
                print(i , j )
        print(ratios, AUCs)
        #plt.errorbar(ratios, np.mean(AUCs,axis=1),yerr=deviation,ecolor='g', color='g',marker='o')
        deviation = np.zeros((2,AUCs.shape[0]))
        deviation[1] = np.max(AUCs,axis=1)
        deviation[0] = np.min(AUCs,axis=1)
        plt.plot(ratios, np.mean(AUCs,axis=1),marker='o',color='g')
        plt.fill_between(ratios, y1=deviation[0],y2=deviation[1],alpha=0.2, edgecolor='#3F7F4C',facecolor='#7EFF99')
        plt.title('AUC for increasing data unbalance (using undersampling)\n')
        plt.ylabel('Area under the curve (AUC)')
        plt.xlabel('Ratio between minority and majority classes (1:X)')
        plt.show()

        #print(ratios)
        #print(AUCs)

if __name__ == '__main__':
   experiment = Experiment(3,sigma=0.00)
   experiment.duplication_experiment()