import numpy as np 
import keras
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras import regularizers

class conv_model():
	# Build model with the given parameters
	def build_model(conv_neurons, conv_repeat, dense_neurons, input_shape, n_class, dropout, regularizer):
		model = Sequential()
		# Conv2D
		for idx, neurons in enumerate(conv_neurons):
			for i in range(conv_repeat):
				if idx == 0:				
					model.add(Conv2D(neurons, (3,3), padding='same',
						input_shape=input_shape[1:], kernel_regularizer=regularizers.l2(regularizer),
						kernel_initializer='he_normal'))					
				else:				
					if i == 0:
						model.add(Conv2D(neurons, (3,3), padding='same', strides=2, 
							kernel_regularizer=regularizers.l2(regularizer), kernel_initializer='he_normal'))
					else:
						model.add(Conv2D(neurons, (3,3), padding='same', 
							kernel_regularizer=regularizers.l2(regularizer), kernel_initializer='he_normal'))
				model.add(BatchNormalization())
				model.add(Activation('relu'))							
		
		pool_size = input_shape[1]/(2**(len(conv_neurons)-1))
		model.add(AveragePooling2D(pool_size=int(pool_size)))
		model.add(Flatten())

		# Dense
		for idx, neurons in enumerate(dense_neurons):			
			model.add(Dense(neurons, kernel_regularizer=regularizers.l2(regularizer), kernel_initializer='he_normal'))
			model.add(Activation('relu'))
			model.add(Dropout(dropout))
		model.add(Dense(n_class, kernel_regularizer=regularizers.l2(regularizer), kernel_initializer='he_normal'))
		model.add(Activation('softmax'))

		model.summary()
		return model 