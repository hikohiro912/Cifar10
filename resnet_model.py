import numpy as np 
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, Dropout
from keras.regularizers import l2
from keras.models import Model

class resnet_model():
	# Conv-BN-Relu-Layer
	def resnet_layer(self, inputs, neurons, kernel_size=3, strides=1, batch=True, activation='relu'):
		conv = Conv2D(neurons, kernel_size=kernel_size, strides=strides, padding='same',
			kernel_initializer='he_normal', kernel_regularizer=l2(1e-2))
		x = inputs
		x = conv(x)
		if batch:
			x = BatchNormalization()(x)
		if activation is not None:
			x = Activation(activation)(x)
		return x

	# Build model with the given parameters
	def build_model(self, x_shape, n_class, stack_depth, block_depth, neuron_0):
		# Init 		
		neurons = neuron_0
		# Input
		inputs = Input(shape=x_shape)
		x = self.resnet_layer(inputs=inputs, neurons=neurons)
		# Resnet
		for stack in range(stack_depth):
			for block in range(block_depth):
				strides = 1
				if stack > 0 and block == 0:
					strides = 2  # reduce dimension if first layer in block
				y = self.resnet_layer(inputs=x, neurons=neurons, strides=strides)
				y = self.resnet_layer(inputs=y, neurons=neurons, activation=None)
				if stack > 0 and block == 0:
					x = self.resnet_layer(inputs=x, neurons=neurons, kernel_size=1,
						strides=strides, activation=None, batch=False)
				x = keras.layers.add([x, y])
				x = Activation('relu')(x)
			neurons *= 2

		# Classifier		
		x = AveragePooling2D(pool_size=int(x.shape[1]))(x)
		y = Flatten()(x)
		y = Dense(256, activation='relu',kernel_initializer='he_normal')(y)
		y = Dropout(0.2)(y)
		outputs = Dense(n_class, activation='softmax', kernel_initializer='he_normal')(y)

		# Create model
		model = Model(inputs=inputs, outputs=outputs)

		model.summary()
		return model 
