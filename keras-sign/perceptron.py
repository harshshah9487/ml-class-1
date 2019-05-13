# A very simple perceptron for classifying american sign language letters
import signdata
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Reshape, Input, Add, ReLU
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.loss = "categorical_crossentropy"
config.optimizer = "adam"
config.epochs = 10

# load data
(X_test, y_test) = signdata.load_test_data()
(X_train, y_train) = signdata.load_train_data()



img_width = X_test.shape[1]
img_height = X_test.shape[2]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]

# you may want to normalize the data here..

# normalize data
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# create model
#model = Sequential()
#model.add(Reshape((img_width, img_height,1), input_shape=(img_width, img_height)))
#model.add(Conv2D(32, (3,3), padding='valid' , activation="relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.35))
#model.add(Conv2D(96, (3,3), padding='valid',activation="relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.35))
#model.add(Flatten())
#model.add(Dropout(0.35))
#model.add(Dense(100, activation="relu"))
#model.add(Dropout(0.35))
#model.add(Dense(50, activation="relu"))
#model.add(Dropout(0.30))
#model.add(Dense(num_classes, activation="softmax"))
#model.compile(loss=config.loss, optimizer=config.optimizer,
#              metrics=['accuracy'])

inp = Input(shape=(img_width, img_height))
reshape_1 = Reshape((img_width, img_height, 1))(inp)
conv_1 = Conv2D(32, (3,3), padding='same', activation='relu')(reshape_1)
#maxpool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)
drop_1 = Dropout(0.35)(conv_1)
conv_2 = Conv2D(96, (3,3), padding='same')(drop_1)
#maxpool_2 = MaxPooling2D(pool_size=(2,2))(conv_2)
#reshape_2 = Reshape((img_width, img_height, 96))(inp)
conv_3 = Conv2D(96, (1,1), padding='same')(reshape_1)
add_1 = Add()([conv_2, conv_3])
relu_1 = ReLU()(add_1)
drop_2 = Dropout(0.35)(relu_1)
flat_1 = Flatten()(drop_2)
drop_3 = Dropout(0.4)(flat_1)
dense_1 = Dense(100, activation="relu")(drop_3)
drop_4 = Dropout(0.4)(dense_1)
dense_2 = Dense(50, activation="relu")(drop_4)
drop_5 = Dropout(0.30)(dense_2)
dense_3 = Dense(num_classes, activation="softmax")(drop_5)
model = Model(inp, dense_3)

model.compile(loss=config.loss, optimizer=config.optimizer,
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test),
          callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])
