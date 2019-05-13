from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import CuDNNLSTM as LSTM
from keras.layers import CuDNNGRU as GRU
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras.datasets import imdb
import wandb
from wandb.keras import WandbCallback
import imdb
import numpy as np
from keras.preprocessing import text

wandb.init()
config = wandb.config

# set parameters:
config.vocab_size = 2000
config.maxlen = 400
config.batch_size = 32
config.embedding_dims = 75
config.filters = 250
config.kernel_size = 3
config.hidden_dims = 100
config.epochs = 13

(X_train, y_train), (X_test, y_test) = imdb.load_imdb()
#print("Before:", X_train[0])
tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
#print("After:", X_train[0])


X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)

model = Sequential()
model.add(Embedding(config.vocab_size,
                    config.embedding_dims,
                    input_length=config.maxlen))
model.add(Dropout(0.35))
model.add(Conv1D(config.filters, config.kernel_size, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(GRU(50, return_sequences=True))#, recurrent_dropout=0.25))
#model.add(GRU(25, return_sequences=True))#, recurrent_dropout=0.25))

model.add(Flatten())

model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(X_test, y_test), callbacks=[WandbCallback()])

model.save("seniment.h5")
