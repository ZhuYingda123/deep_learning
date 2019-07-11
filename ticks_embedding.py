from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Dropout
from keras.callbacks import EarlyStopping
from data_lab import *
import matplotlib.pyplot as plt

path = '/data/dataDisk2/yingda/'
path2 = '/data/dataDisk2/yingda/return'

def process_line(line):
    tmp = [int(val) for val in line.strip().split(',')]
    x = np.array(tmp[:-1])
    y = np.array(tmp[-1:])
    return x,y

def generate_arrays_from_files(path, path2, day_begin, day_end, batch_size):
    day_list = estab_day_list(day_begin,day_end)
    cnt = 0
    X =[]
    Y =[]
    for date in day_list:
        path_data = path+str(date)+'.npy'
        if not os.path.exists(path_data):
            continue
        array_data = np.load(path_data)
        label_data = get_lab(path2, start=str(date), end = str(date)).values
        codes_index1 = label_data[:, 0] #array
        label_data = label_data[:, 1]
        codes_index2 = pd.read_csv(code_index_file, dtype={'stock_code':str}).set_index('stock_code')['idx'] #series
        # num_codes = array_data.shape[0]
        for j, code in enumerate(codes_index2.keys()):
            # print(code)
            if not code in codes_index1:
                continue
            X.append(array_data[j].reshape((15, 4801)).T)
            Y.append(label_data[int(np.argwhere(codes_index1 == code))])
            # Y.append(label_data[j])
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                yield (np.array(X), np.array(Y))
                X = []
                Y = []

def generate_training_data(text_like_data, max_text_size):
    text_num = len(text_like_data)
    train_x = []
    train_y = []
    for i in range(text_num):
        text_size = len(text_like_data[i])
        if text_size <= 3*max_text_size/4:
            continue
        if text_size<max_text_size \
            and text_size>3*max_text_size/4:
            for k in range(max_text_size-text_size):
                text_like_data[i].append(text_like_data[i][text_size-1])
        mid = int(max_text_size/2)
        y = text_like_data[i][mid]
        x = text_like_data[i][0]
        for j in range(1,max_text_size):
            if j != mid:
                # x = np.vstack((x, text_like_data[i][j]))
                x = np.hstack((x, text_like_data[i][j]))
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

def trans_to_low_dim(x, W, alpha=0.3):
    x_t = x.dot(W)
    x_t[np.nonzero(x_t[:]<0)[0]]/=0.3
    return x_t

class embedding_model:
    def __init__(self):
        self.model = Sequential()
        self.weight = None

    def establish(self, x_dim, y_dim):
        self.model.add(Dense(32, input_dim=x_dim, name='dense_1'))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dense(units=y_dim, name='dense_2'))
        # self.model.add(LeakyReLU(alpha=0.05))
        self.model.compile(loss='mse',
              optimizer='Adam',
              metrics=['mse'])
        
    def train(self, train_x, train_y):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit(train_x, train_y, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping])
        # self.model.save_weights('my_model_weights.h5')
        self.weight = self.model.get_weights()[2].T
    
    def predict(self, test_x):
        return self.model.predict(test_x)

class conv1D_model:
    def __init__(self):
        self.model = Sequential()
        
    def establish(self, tick_len, x_dim):
        self.model.add(Conv1D(64, 32, activation='relu', input_shape=(tick_len, x_dim)))
        self.model.add(Conv1D(64, 32, activation='relu'))
        self.model.add(MaxPooling1D(3))
        self.model.add(Conv1D(128, 32, activation='relu'))
        self.model.add(Conv1D(128, 32, activation='relu'))
        self.model.add(MaxPooling1D(3))
        self.model.add(Conv1D(128, 16, activation='relu'))
        self.model.add(Conv1D(128, 16, activation='relu'))
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1))

        self.model.compile(loss='mse',
                    optimizer='rmsprop',
                    metrics=['mse'])
        print(self.model.summary())

    def train(self, generator):
        # early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit_generator(generator, steps_per_epoch=100, epochs=100, verbose=1, callbacks=None, validation_data=None, \
            validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, initial_epoch=0)
        self.model.save_weights('my_model_weights.h5')
        # self.model.fit_generator(generator, steps_per_epoch=10, epochs=100)
        print(self.model.get_weights())


if __name__ == '__main__':
    '''
    tag = 'ask1'
    day_list = [20160503, 20160504, 20160505,20160506,20160507, 20160508, 20160509, 20160510, 20160511, 20160512, 20160513, 20160514, 20160515, 20160516, 20160517, 20160518, 20160519, 20160520, 20160521, 20160522, 20160523, 20160524]
    one_day_data = estab_one_day_data_lab(tag, day_list, is_chafen = False, is_zero_mean = False, is_minus_open = True)
    text_like_data = estab_text_like_data_lab(one_day_data, 3)
    x,y = generate_training_data(text_like_data, 3)
    x_train = x[:-5, :]
    y_train = y[:-5, :]
    embedding = embedding_model()
    embedding.establish(4802*2, 4802)
    embedding.train(x_train,y_train)
    '''
    # weight = np.zeros((4802, 32))
    # for i in range(32):
    #     weight[i,i]=1
    # print(len(weight))
    # model = embedding.model.load_weights('my_model_weights.h5')
    # np.savetxt('weight.txt',weight)
    '''
    for i in range(3):
        x_orig = text_like_data[i][0]
        x_trans = trans_to_low_dim(x_orig, embedding.weight)
        print(x_trans)
        plt.figure(1, figsize=(15, 10))
        plt.plot(np.arange(4802),x_orig,c='blue')
        plt.title(tag+'_'+str(i),fontsize = 40)
        plt.savefig(tag+'_'+str(i))
        plt.close(1)
        plt.figure(2, figsize=(15, 10))
        plt.plot(np.arange(32),x_trans,c='blue')
        plt.title(tag+'_'+str(i)+'_trans',fontsize = 40)
        plt.savefig(tag+'_'+str(i)+'_trans')
        plt.close(2)

    plt.figure(1, figsize=(15, 10))
    plt.plot(np.arange(4802),y[-2, :],c='red')
    plt.title(tag+'_'+'real',fontsize = 40)
    plt.savefig(tag+'_real')
    plt.close(1)
    y_pre = embedding.predict(x[-2:-1, :])
    print(y_pre)
    plt.figure(2, figsize=(15, 10))
    plt.plot(np.arange(4802),y_pre[0, :],c='red')
    plt.title(tag+'_'+'pre',fontsize = 40)
    plt.savefig(tag+'_pre')
    plt.close(2)
    '''
    # generate_arrays_from_files(path, path2, 20190401, 20190402, 32)
    generator = generate_arrays_from_files(path, path2, 20190401, 20190412, 32)
    conv1d = conv1D_model()
    conv1d.establish(4801,15)
    conv1d.train(generator)