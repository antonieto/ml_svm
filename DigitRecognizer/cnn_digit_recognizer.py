import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import os

train = pd.read_csv('./test.csv')
test = pd.read_csv('./test.csv')

def main():
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
 
    X_train /= 255
    X_test /= 255

    
    n_classes = 10
    print("Shape before one-hot encoding: ", y_train.shape)
    Y_train = to_categorical(y_train, n_classes)
    Y_test = to_categorical(y_test, n_classes)

    model = Sequential()

    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))                            
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # Train and save model
    # training the model and saving metrics in history
    
    # saving the model
    model_name = 'keras_mnist.h5'
    
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Specify the relative path from the script's directory (replace 'my_model.h5' with your desired filename)
    save_path = os.path.join(script_dir, model_name)

    # Save the model to the specified path
    model.save(save_path)
    
if __name__ == '__main__':
    main()
