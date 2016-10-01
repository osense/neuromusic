import numpy as np
import time
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.layers.recurrent import LSTM
import neuralio
import datahandler


class Net:
    model = None
    dropout = 0.2

    def new(self):
        a = Input(shape=(10000,3))
        b = LSTM(input_dim=10000, output_dim=2000, return_sequences=True, dropout_U=self.dropout, dropout_W=self.dropout)(a)
        c = LSTM(input_dim=2000, output_dim=1000, return_sequences=False, dropout_U=self.dropout, dropout_W=self.dropout)(b)
        z = Dense(input_dim=1000, output_dim=10000)(c)
        self.model= Model(input=a, output=z)

        start = time.time()
        self.model.compile(loss="mse", optimizer="rmsprop")
        print "Compilation Time : ", time.time() - start

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
