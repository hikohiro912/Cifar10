import pickle
import numpy as np 
import keras 

class dataProcessor():
	def __init__(self, files):
		self.files = files 
	def getData(self, n_class):
		X = []; Y = []
		for file in self.files:
			with open(file, 'rb') as fo:
				d = pickle.load(fo, encoding='bytes')
			x, y = self.shuffleDict(d)
			X.extend(x)
			Y.extend(y)
		X = np.array(X).astype('float32')
		X /= 255
		Y = keras.utils.to_categorical(np.array(Y), n_class)
		return np.array(X), np.array(Y)
		
	def shuffleDict(self, d):
		x = d.get(b'data')
		y = d.get(b'labels')
		y = np.array(y)
		# Reshape x		
		x_tmp = []
		for im in x:
			im_tmp = np.reshape(im, (3, -1)).transpose()
			x_tmp.append(np.reshape(im_tmp, (32, 32, 3)))		
		x = np.array(x_tmp)		
		
		# Shuffle
		n_data = x.shape[0]
		arr=np.arange(n_data)
		np.random.shuffle(arr)
		x = x[arr]
		y = y[arr]
		
		return x, y

