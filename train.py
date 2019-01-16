from dataProcessor import dataProcessor
from conv_model import conv_model 
from resnet_model import resnet_model 
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
import os, time
import matplotlib.pyplot as plt

####### Parameters #######
# General
n_class = 10
# Files
train_data_files = ['data/data_batch_1','data/data_batch_2','data/data_batch_3',
	'data/data_batch_4','data/data_batch_5']
test_data_files = ['data/test_batch']
save_dir = os.path.join(os.getcwd(), 'saved_models')
# Conv Model
conv_neurons = [32, 64, 128]
conv_repeat = 2
dense_neurons = [512]
dropout = 0.2
regularizer = 0.01
# Resnet model
stack_depth = 3
block_depth = 3
neurons_0 = 16
# Training
lr = 0.0001
batch_size = 32
epoch = 20
data_gen = False
isResnet = False

####### Data #######
print('Getting Data...', end='')
train_data_processor = dataProcessor(train_data_files)
x_train, y_train = train_data_processor.getData(n_class)
test_data_processor = dataProcessor(test_data_files)
x_test, y_test = test_data_processor.getData(n_class)
print('Done')

####### Model #######
if isResnet:
	Resnet_model = resnet_model()
	model = Resnet_model.build_model(x_train.shape[1:], n_class, stack_depth, block_depth, neurons_0)
else:
	model = conv_model.build_model(conv_neurons, conv_repeat, dense_neurons, x_train.shape, 
		n_class, dropout, regularizer)

opt = Adam(lr=lr, decay=1e-6)
model.compile(loss='categorical_crossentropy', 
	optimizer=opt, metrics=['accuracy'])

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
if isResnet:
	model_name = 'res_{}x{}x{}-{}.h5'.format(stack_depth, block_depth, neurons_0, int(time.time()))
else:
	model_name = 'conv_{}x{}_{}-{}.h5'.format(str(conv_neurons), conv_repeat, str(dense_neurons), int(time.time()))
model_path = os.path.join(save_dir, model_name)

# Callbacks
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_acc', verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
callbacks = [checkpoint, tensorboard]

# Train
if data_gen:
	datagen = ImageDataGenerator(
	        featurewise_center=False,  # set input mean to 0 over the dataset
	        samplewise_center=False,  # set each sample mean to 0
	        featurewise_std_normalization=False,  # divide inputs by std of the dataset
	        samplewise_std_normalization=False,  # divide each input by its std
	        zca_whitening=False,  # apply ZCA whitening
	        zca_epsilon=1e-06,  # epsilon for ZCA whitening
	        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
	        # randomly shift images horizontally (fraction of total width)
	        width_shift_range=0.1,
	        # randomly shift images vertically (fraction of total height)
	        height_shift_range=0.1,
	        shear_range=0.,  # set range for random shear
	        zoom_range=0.,  # set range for random zoom
	        channel_shift_range=0.,  # set range for random channel shifts
	        # set mode for filling points outside the input boundaries
	        fill_mode='nearest',
	        cval=0.,  # value used for fill_mode = "constant"
	        horizontal_flip=True,  # randomly flip images
	        vertical_flip=False,  # randomly flip images
	        # set rescaling factor (applied before any other transformation)
	        rescale=None,
	        # set function that will be applied on each input
	        preprocessing_function=None,
	        # image data format, either "channels_first" or "channels_last"
	        data_format=None,
	        # fraction of images reserved for validation (strictly between 0 and 1)
	        validation_split=0.0)
	datagen.fit(x_train)
	history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
		epochs=epoch, validation_data=(x_test, y_test), workers=4, callbacks=callbacks)
else:
	history = model.fit(x_train, y_train, batch_size=batch_size,
		epochs=epoch, validation_data=(x_test, y_test), shuffle=True, callbacks=callbacks)

# Load best model
model = load_model(model_path)

# Score trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model_name = '[{}]{}'.format(scores[1]*100, model_name)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Final model saved to %s' % model_path)

# Predict 
class_list = ['airplane','automobile','bird','cat','deer',
	'dog','frog','horse','ship','truck']
n_prediction = 5
for predict in range(n_prediction):
	# Do prediction
	rand_index = random.rand(0, x_test.shape[0]-1)
	x_predict = x_test[rand_index]
	y_predict = model.predict(x_predict)
	class_predict = class_list[np.argmax(y_predict)]
	# Plot the figure with predicted title
	plt.imshow(x_predict)
	plt.title(class_predict)
	plt.show()
	


