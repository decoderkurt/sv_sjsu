from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
import keras.backend.tensorflow_backend as K
import numpy

print(device_lib.list_local_devices())

# random seed for reproducibility
numpy.random.seed(2)


# loading load prima indians diabetes dataset, past 5 years of medical history 
dataset = numpy.loadtxt("Churn_Modelling_2.csv", delimiter=",")

# split into input (X) and output (Y) variables, splitting csv data
X = dataset[:,0:11]
Y = dataset[:,11]

# split X, Y into a train and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

with K.tf.device('/gpu:0'):
    # create model, add dense layers one by one specifying activation function
    model = Sequential()
    model.add(Dense(15, input_dim=11, activation='relu')) # input layer requires input_dim param
    model.add(Dense(10, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(1, activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1

    # compile the model, adam gradient descent (optimized)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    # call the function to fit to the data (training the network)
    model.fit(x_train, y_train, epochs = 80, batch_size=100, validation_data=(x_test, y_test))

    # save the model
    model.save('weights.h5')

    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))