from dataProcessor import dataProcessor
from keras.models import load_model
import os, random 
import matplotlib.pyplot as plt
import numpy as np 

####### Parameters #######
# General
n_class = 10
# Files
data_files = ['data/data_batch_1','data/data_batch_2','data/data_batch_3',
	'data/data_batch_4','data/data_batch_5', 'data/test_batch']
save_dir = os.path.join(os.getcwd(), 'saved_models')
# Model 
model_name = '[97.01][80.80]res_5x3x16-1547728560.h5'
model_path = os.path.join(save_dir, model_name)
model = load_model(model_path)

####### Data #######
batch_size = 32
print('Getting Data...', end='')
data_processor = dataProcessor(data_files)
x, y = data_processor.getData(n_class)
print('Done')

####### Predict #######
class_list = ['airplane','automobile','bird','cat','deer',
	'dog','frog','horse','ship','truck']
n_prediction = 10
for predict in range(n_prediction):
	# Do prediction
	rand_index = random.randint(0, x.shape[0]-1)
	x_predict = np.array([list(x[rand_index])]*batch_size)
	y_predict = model.predict(x_predict, batch_size=batch_size)
	class_predict = class_list[np.argmax(y_predict[0])]
	# Plot the figure with predicted title
	plt.imshow(x_predict[0])
	plt.title(class_predict)
	plt.show()
	


