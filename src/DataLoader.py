# Summarize the Credit Card Fraud dataset
import numpy as np 
from pandas import read_csv


class DataLoader:
	def __init__(self, datafile='creditcard.csv'):
		path = 'data/' + datafile
		# load the dataset
		dataframe = read_csv(path)
		# get the values
		values = dataframe.values
		self.X, self.Y = values[:, :-1], values[:, -1]
		mean = np.mean(self.X)
		var = np.var(self.X)
		#self.X = (self.X-mean)/np.sqrt(var)
		min = np.max(self.X)
		max = np.min(self.X)
		self.X = (self.X-min)/(max-min)
		# gather details
		
	
	def summarize(self):
		n_rows = self.X.shape[0]
		n_cols = self.X.shape[1]
		classes = np.unique(self.Y)
		n_classes = len(classes)
		# summarize
		print('N Examples: %d' % n_rows)
		print('N Inputs: %d' % n_cols)
		print('N Classes: %d' % n_classes)
		print('Classes: %s' % classes)
		print('Class Breakdown:')
		# class breakdown
		breakdown = ''
		for c in classes:
			total = len(self.Y[self.Y == c])
			ratio = (total / float(len(self.Y))) * 100
			print(' - Class %s: %d (%.5f%%)' % (str(c), total, ratio))