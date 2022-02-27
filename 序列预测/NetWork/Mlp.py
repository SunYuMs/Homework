from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM,Dropout
from Network import NeuralNetwork
from Network import Layer

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

def sMAPE(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred+y_true) )) * 100

if __name__ == '__main__':
    ##参数
    n_input=Data.Input_length   ##输入长度
    n_output=56
    learning_rate=0.01
    max_epochs=1000
    bath_size=20
    Loss_type = 'MSE'  # MSE、MAE
    Rate_type = 'fixed'  # exponent、preheat、fixed
    Regularization_type = 'none'  # l1、l2、none
    alpha = 0.0001  #正则化参数
    beta = 0.1      #动量SGD参数
    #####数据处理
    train_X = Data.train_X
    train_y = Data.train_y
    test_X = Data.train_X.reshape(-1, 111, n_input)
    test_X=test_X[-1].reshape(111, n_input)
    test_y = Data.test

    ###网络
    layer_1 = Layer(n_input, n_input, 'sigmoid')  # softmax、sigmoid、tanh、relu
    layer_2 = Layer(n_input, n_output, 'sigmoid')
    nn = NeuralNetwork()
    nn.add_layer(layer_1)
    nn.add_layer(layer_2)

    ##train
    train_loss,train_smape,eva_loss,eva_smape,test_loss,test_smape=nn.train(train_X, test_X, train_y, test_y,
                                                        learning_rate, max_epochs, bath_size, Loss_type,
                                                        Rate_type, Regularization_type, alpha,beta)
    ###out
    print("end \ntrain_loss %f : train_smape %.2f \neva_loss %f : eva_smape %.2f \ntest_loss %f : test_smape %.2f\nend"
          %(train_loss,train_smape,eva_loss,eva_smape,test_loss,test_smape))






