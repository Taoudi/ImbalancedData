# Summarize the Credit Card Fraud dataset
import numpy as np 
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:
	def __init__(self, datafile='creditcard.csv'):
		scaler = StandardScaler()

		#Convert to numpy arrays for tensorflow
		#self.X = scaler.fit_transform(X_train,y_train)
		#self.testX = scaler.transform(X_test)


		path = 'data/' + datafile
		dataframe = read_csv(path)
		values = dataframe.values
		self.X, self.Y = values[:, :-1], values[:, -1]
		mean = np.mean(self.X)
		var = np.var(self.X)
		self.min = np.max(self.X)
		self.max = np.min(self.X)
		#self.X = (self.X-self.min)/(self.max-self.min)
		#print(np.std(self.X,axis=0))

		self.X, self.testX, self.Y, self.testY = train_test_split(self.X,self.Y, test_size=0.15)
		self.X, self.valX, self.Y, self.valY = train_test_split(self.X,self.Y, test_size=0.1)
		self.X = scaler.fit_transform(self.X,self.Y)
		self.valX = scaler.fit_transform(self.valX,self.valY)

		self.testX = scaler.transform(self.testX)
		print(np.std(self.X,axis=0)) #Check that the column standard deviations are all 1

		self.classes = np.unique(self.Y)
		self.n_classes = len(self.classes)

	def normalize(self):
		self.X = (self.X-self.min)/(self.max-self.min)
		self.valX =  (self.valX-self.min)/(self.max-self.min)
		self.testX = (self.testX-self.min)/(self.max-self.min)
		

	def summarize(self,test=False):
		if test:
			X = self.testX
			Y = self.testY
		else:
			X = self.X
			Y = self.Y
		n_rows = self.X.shape[0]
		n_cols = self.X.shape[1]
		classes = np.unique(Y)
		n_classes = len(classes)
		# summarize
		print('N Classes: %d' % n_classes)
		print('Classes: %s' % classes)
		print('Class Breakdown:')
		# class breakdown
		breakdown = ''
		for c in classes:
			total = len(Y[Y == c])
			ratio = (total / float(len(Y))) * 100
			print(' - Class %s: %d (%.5f%%)' % (str(c), total, ratio))