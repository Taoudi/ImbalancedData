# Summarize the Credit Card Fraud dataset
import numpy as np 
from pandas import read_csv
from sklearn.model_selection import train_test_split

class DataLoader:
	def __init__(self, datafile='creditcard.csv'):
		path = 'data/' + datafile
		dataframe = read_csv(path)
		values = dataframe.values
		self.X, self.Y = values[:, :-1], values[:, -1]
		mean = np.mean(self.X)
		var = np.var(self.X)
		min = np.max(self.X)
		max = np.min(self.X)
		self.X = (self.X-min)/(max-min)
		self.X, self.testX, self.Y, self.testY = train_test_split(self.X,self.Y, test_size=0.15)
		self.X, self.valX, self.Y, self.valY = train_test_split(self.X,self.Y, test_size=0.12)
		self.classes = np.unique(self.Y)
		self.n_classes = len(self.classes)

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