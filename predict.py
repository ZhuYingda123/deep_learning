from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU, Dropout, Embedding, LSTM
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from data_lab import *
import matplotlib.pyplot as plt

def estab_squen_training_data(text_like_data, return_data, weight, max_text_size):
    text_num = len(text_like_data)
    train_x = []
    train_y = []
    for i in range(text_num):
        text_size = len(text_like_data[i])
        if text_size < max_text_size:
            continue
        x = []
        for j in range(max_text_size):
            x.append(text_like_data[i][j].dot(weight))
        y = return_data[i+max_text_size]
        # x /= max_text_size-1
        # print(y.shape)
        # print(x.shape)
        train_y.append(y)
        train_x.append(x)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    # train_x = (train_x-min_*np.ones((train_x.shape[0], train_x.shape[1])))/(max_-min_)
    # train_y = (train_y-min_*np.ones((train_x.shape[0], train_x.shape[1])))/(max_-min_)
    # np.savetxt('x.txt',train_x)
    # np.savetxt('y.txt', train_y)
    # print(train_x)
    # print(train_y)
    
    return train_x, train_y

class return_predict:
    def __init__(self):
        self.model = Sequential()
        self.max_features = 1024

    def establish(self):
        # self.model.add(Embedding(self.max_features, output_dim=2)) 
        # keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
        self.model.add(LSTM(128, input_shape=(1,10))) # 128 is the num of neurons in the hidden layer
        # model.add(LSTM(50, input_shape=(timestep, dim)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train, batch_size=16, epochs=10)

    def predict(self, x_test):
        return self.model.predict(x_test)

if __name__ == '__main__':
    x_train = np.random.rand(500,10)
    print(x_train)
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    print(x_train)
    x_test = np.random.rand(100, 10)
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    y_train = to_categorical(np.random.randint(0,2,size =(500,1)))
    print(y_train.shape)
    y_test = to_categorical(np.random.randint(0,2,size =(100,1)))
    predict_model = return_predict()
    predict_model.establish()
    predict_model.train(x_train, y_train)
    print(predict_model.predict(x_test))
