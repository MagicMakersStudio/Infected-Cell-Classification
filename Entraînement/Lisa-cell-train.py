
#-=-=-=-=-=-=-=-=-=-=-#
#       IMPORT        #
#-=-=-=-=-=-=-=-=-=-=-#

import keras
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import h5py
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#-=-=-=-=-=-=-=-=-=-=-#
#        DATA         #
#-=-=-=-=-=-=-=-=-=-=-#

Parasitized = glob("../Data/cell_images/Parasitized/*.png") # entre guillemets pour dire que l'on cherche une chaîne de caractères
Uninfected = glob("../Data/cell_images/Uninfected/*.png") # entre guillemets pour dire que l'on cherche une chaîne de caractères

x_train = []
y_train = []

label_infected = 0
label_uninfected = 1

for image in tqdm(Parasitized):
	img = Image.open(image).convert("L").resize((100, 100))
	img = np.array(img)
	img = img.reshape(100, 100)
	img = img.astype('float32')
	img /= 255

	x_train.append(img)
	y_train.append(label_infected)

for image in tqdm(Uninfected):
	img = Image.open(image).convert("L").resize((100, 100))
	img = np.array(img)
	img = img.reshape(100, 100)
	img = img.astype('float32')
	img /= 255

	x_train.append(img)
	y_train.append(label_uninfected)


x_train = np.array(x_train)
x_train = x_train.reshape(27518,100,100,1)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1)

print(x_train.shape)
print(x_test.shape)
print(x_train[0])
print(y_train[0])

y_train = to_categorical(y_train,2)
y_test = to_categorical(y_test, 2)

print(x_train[0])
print(y_train[0])

#-=-=-=-=-=-=-=-=-=-=-#
#       MODÈLE        #
#-=-=-=-=-=-=-=-=-=-=-#

model = Sequential()

model.add(Conv2D(32, kernel_size = 3, input_shape = (100, 100, 1)))
model.add(LeakyReLU(0.2))

model.add(Conv2D(64, kernel_size = 3))
model.add(LeakyReLU(0.2))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, kernel_size = 3))
model.add(LeakyReLU(0.2))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(32, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
	optimizer = Adam(),
	metrics=['accuracy'])

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#       TRAIN - ENTRAÎNEMENT        #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

model.fit(x_train, y_train,
	batch_size = 30,
	epochs = 10,
	verbose = 1,
	validation_data = (x_test, y_test))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#       SAUVEGARDER MODÈLE        #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

model.save("model.h5")
