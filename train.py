from dataProcessor import dataProcessor
from conv_model import conv_model 
from keras.optimizers import Adam

####### Parameters #######
# General
n_class = 10
# Files
train_data_files = ['data/data_batch_1','data/data_batch_2','data/data_batch_3',
	'data/data_batch_4','data/data_batch_5']
test_data_files = ['data/test_batch']
# Model
conv_neurons = [32, 64, 128]
conv_repeat = 3
dense_neurons = [512]
dropout = 0.2
regularizer = 0.01
# Training
lr = 0.001
batch_size = 32
epoch = 20

####### Data #######
print('Getting Data...', end='')
train_data_processor = dataProcessor(train_data_files)
x_train, y_train = train_data_processor.getData(n_class)
test_data_processor = dataProcessor(test_data_files)
x_test, y_test = test_data_processor.getData(n_class)
print('Done')

####### Model #######
model = conv_model.build_model(conv_neurons, conv_repeat, dense_neurons, x_train.shape, 
	n_class, dropout, regularizer)
opt = Adam(lr=lr)
model.compile(loss='categorical_crossentropy', 
	optimizer=opt, metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size,
	epochs=epoch, validation_data=(x_test, y_test), shuffle=True)
