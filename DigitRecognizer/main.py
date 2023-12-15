from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical

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
    Y_test = to_categorical(y_test, n_classes)
    
    mnist_model = load_model("./keras_mnist.h5")
    loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)

    print("Test Loss", loss_and_metrics[0])
    print("Test Accuracy", loss_and_metrics[1])
    
if __name__ == '__main__':
    main()
