import numpy as np 
import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras import regularizers

class conv_model():
	# Build model with the given parameters
	def build_model(conv_neurons, dense_neurons, input_shape, n_class, dropout, regularizer):
		model = Sequential()
		# Conv2D
		for idx, neurons in enumerate(conv_neurons):
			if idx == 0:
				model.add(Conv2D(neurons, (3,3), padding='same',
					input_shape=input_shape[1:]))
			else:
				model.add(Conv2D(neurons, (3,3), padding='same'))
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2,2)))
			model.add(Dropout(dropout))

		model.add(Flatten())

		# Dense
		for idx, neurons in enumerate(dense_neurons):
			if idx == len(dense_neurons) - 1:
				model.add(Dense(n_class, 
					kernel_regularizer=regularizers.l2(regularizer)))
				model.add(Activation('softmax'))
			else:
				model.add(Dense(neurons, kernel_regularizer=regularizers.l2(regularizer)))
				model.add(Activation('relu'))
				model.add(Dropout(dropout))

		model.summary()
		return model 