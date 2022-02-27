from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM,Dropout
import numpy as np
import  Data
from sklearn.preprocessing import MinMaxScaler
def seed(seed_value):
    # Seed value
    # Apparently you may use different seed values at each stage
    # seed_value= 0

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.random.set_seed(seed_value)
    # for later versions:
    # tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    from keras import backend as K
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)
    # for later versions:
    # session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # tf.compat.v1.keras.backend.set_session(sess)

seed(1)

def MLP(n_input, n_output, X, y, epochs_num):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=n_input))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs_num, verbose=0)
    return model

def Lstm(n_input, n_output, X, y, epochs_num):
    model = Sequential()
    model.add(LSTM(100,  input_shape=(n_input, 1)))
    model.add(Dense(n_output))
    # model.add(LSTM(100, return_sequences=True, input_shape=(60, 1)))
    # model.add(LSTM(100))
    # model.add(Dropout(0.2))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs_num,verbose=0)
    return model

def sMAPE(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred+y_true) )) * 100

if __name__ == '__main__':
    ##参数
    epochs_num = 2
    n_input=60
    n_output=56
    ##数据处理
    X = Data.train_X
    y = Data.train_y
    scaler_train_X  = MinMaxScaler(feature_range=(0, 1))
    scaler_train_y = MinMaxScaler(feature_range=(0, 1))
    X = scaler_train_X.fit_transform(X)
    y = scaler_train_y.fit_transform(y)
    train_X=X.reshape(-1,60,1)
    train_y=y.reshape(-1,56,1)

    ##模型
    model=Lstm(n_input, n_output, train_X, train_y, epochs_num)



    ###train
    X = Data.train_X.reshape(-1,111,60)
    y = Data.train_y.reshape(-1,111,56)
    train_loss=[]
    train_smape=[]
    for i in range(len(X)):
        scaler_train_X = MinMaxScaler(feature_range=(0, 1))
        scaler_train_y= MinMaxScaler(feature_range=(0, 1))
        tx = scaler_train_y.fit_transform(X[i])
        ty = scaler_train_y.fit_transform(y[i])
        tx = model.predict(tx.reshape(111,60,1), verbose=0)
        train_loss_tmp = np.mean(np.square(ty - tx))
        tx = scaler_train_y.inverse_transform(tx)
        train_smape_tmp = sMAPE(y[i], tx)
        train_loss.append(train_loss_tmp)
        train_smape.append(train_smape_tmp)
    print("train loss %f : smape %.3f" % (np.mean(train_loss), np.mean(train_smape)))

    ###var



    ###test
    TX = Data.train_X.reshape(-1, 111, 60)
    scaler_test_X = MinMaxScaler(feature_range=(0, 1))
    scaler_test_y = MinMaxScaler(feature_range=(0, 1))
    TX = scaler_test_X.fit_transform(TX[-1].reshape(111, 60))
    TX = TX.reshape(111, 60, 1)

    y_true = Data.test.reshape(-1, 56)
    y_true = scaler_test_y.fit_transform(y_true)
    y_hat = model.predict(TX, verbose=1)

    test_loss = np.mean(np.square(y_true - y_hat))
    y_hat = scaler_test_y.inverse_transform(y_hat)
    test_smape = sMAPE(Data.test, y_hat)

    print("test loss %f : smape %.3f" % (test_loss, test_smape))



