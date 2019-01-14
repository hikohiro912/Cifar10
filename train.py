from dataProcessor import dataProcessor
from conv_model import conv_model 
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import os

####### Parameters #######
# General
n_class = 10
# Files
train_data_files = ['data/data_batch_1','data/data_batch_2','data/data_batch_3',
	'data/data_batch_4','data/data_batch_5']
test_data_files = ['data/test_batch']
save_dir = os.path.join(os.getcwd(), 'saved_models')
# Model
conv_neurons = [32, 64, 64]
conv_repeat = 2
dense_neurons = [128]
dropout = 0.2
regularizer = 0.01
# Training
lr = 0.0001
batch_size = 64
epoch = 5
data_gen = False

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
		epochs=epoch, validation_data=(x_test, y_test), workers=4)
else:
	history = model.fit(x_train, y_train, batch_size=batch_size,
		epochs=epoch, validation_data=(x_test, y_test), shuffle=True)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_name = '[%3.1f]cifar10-conv-%sx%d-dense-%s.h5' % (scores[1]*100, str(conv_neurons), conv_repeat, str(dense_neurons))
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

