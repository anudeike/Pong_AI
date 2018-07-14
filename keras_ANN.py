import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.metrics import categorical_crossentropy
from keras import regularizers
import pandas as pd
import matplotlib.pyplot as plt

class keras_NN:

    #constructor
    def __init__(self,samples, labels):
        self.samples = samples
        self.labels = labels
        self.scaling()
        self.scaling_MeanNorm()

    @property
    #getters and setters
    def get_samples(self):
        return self.samples

    @property
    def get_labels(self):
        return self.labels

    def get_scaledSample(self):
        return self.scaled_samples

    #setters
    def set_samples(self,samples):
        self.samples = samples

    def set_labels(self, labels):
        self.labels = labels

    def scaling(self):

        nSamples = np.array(self.samples)

        #create object where feature range is between 0,1 (easier)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_Samples = scaler.fit_transform((nSamples))
        return scaled_Samples

    def scaling_MeanNorm(self):
        nSamples = np.array(self.samples)

        #calculate the z score for everything
        mean = np.mean(nSamples)
        stddev = np.std(nSamples)
        scaled_Samples = np.array([(x-mean)/stddev for x in nSamples])
        return scaled_Samples

    def implement_model(self):

        sSamples = self.scaling_MeanNorm()
        aLabels = self.labels
        #creating the model
        model = Sequential([
                            Dense(4, input_shape=(3,), kernel_regularizer=regularizers.l1(0.01), activity_regularizer=regularizers.l2(0.01), activation='relu'),
                            Dense(2,activation='softmax')])

        #complile model
        model.compile(optimizer=SGD(lr=.001, momentum=0.0, ), loss='sparse_categorical_crossentropy', metrics=["accuracy"])

        #train the model
        history = model.fit(sSamples,aLabels, validation_split=0.1, batch_size=700, epochs=100, shuffle=True,verbose=1)

        #plot this data
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig('diagnosis')
        model.save('first_model_simplify_v2.h5')



#TESTING
pathX = r"test_x - Copy.csv"
pathY = r"test_y"

#csv to dataframe
df_samples = pd.read_csv(pathX)
df_labels = pd.read_csv(pathY)

#turn into arrays
aSamples = df_samples.values
aLabels =  df_labels.values

model_0 = keras_NN(aSamples,aLabels)

scaled = model_0.scaling()

model_0.implement_model()

