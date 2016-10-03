import numpy as np
import time
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import neuralio
import datahandler


class Net:
    model = None
    layers = [2000, 1000, 500, 2000]
    dropout = 0.2

    def new(self):
        self.model = Sequential()

        self.model.add(LSTM(
            input_dim=self.layers[0],
            output_dim=self.layers[1],
            return_sequences=True))
        self.model.add(Dropout(self.dropout))

        self.model.add(LSTM(
            output_dim=self.layers[2],
            return_sequences=False))
        self.model.add(Dropout(self.dropout))

        self.model.add(Dense(
            output_dim=self.layers[3]))
        self.model.add(Activation("linear"))

        self.model.compile(loss="mse", optimizer="rmsprop")

    def save(self, path='model.h5'):
        self.model.save(path)

    def load(self, path='model.h5'):
        self.model = load_model(path)

    def learn(self, (X_train, Y_train)):
        start = time.time()
        try:
            self.model.fit(
                X_train, Y_train,
                batch_size=8, nb_epoch=1, validation_split=0.05, verbose=1)
        except KeyboardInterrupt:
            print 'Training duration (s) : ', time.time() - start

    def eval(self, input):
        return self.model.predict(input, batch_size=1)
