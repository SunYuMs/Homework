import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Layer:
    # 全链接网络层
    def __init__(self, n_input, n_output, activation_fun=None,Initialization=None,weights=None, bias=None):
        """
        :param int n_input: 输入节点数
        :param int n_output: 输出节点数
        :param str activation: 激活函数类型
        :param weights: 权值张量，默认类内部生成
        :param bias: 偏置，默认类内部生成
        """
        if Initialization is None :#标准正太分布初始化
            self.weights = weights if weights is not None else np.random.randn(n_input, n_output)
            self.bias = bias if bias is not None else np.random.randn(1,n_output)
        elif Initialization=='he_normal': #正态化的kaiming初始化——he_normal
            self.weights = weights if weights is not None else np.random.normal(loc=0.0, scale=np.sqrt(2 / n_input),
                                                                                size=(n_input, n_output))
            self.bias = bias if bias is not None else np.random.normal(loc=0.0, scale=np.sqrt(2 / n_input),
                                                                       size=(1, n_output))
        elif Initialization=='glorot_normal':#正态化的Glorot初始化——glorot_normal
            self.weights = weights if weights is not None else np.random.normal(loc=0.0, scale=np.sqrt(2 / (n_input+n_output)),
                                                                                size=(n_input, n_output))
            self.bias = bias if bias is not None else np.random.normal(loc=0.0, scale=np.sqrt(2 / (n_input+n_output)),
                                                                   size=(1, n_output))


        self.activation_fun = activation_fun  # 激活函数类型，如’sigmoid’
        self.output = None  # 激活函数的输出值 o

        self.delta_cur = None  # 记录当前层的 delta 变量，用于计算梯度

        self.delta_pre = 0  # 记录当前层的 delta 变量，用于计算梯度

    def compute(self, X):
        # 前向计算函数

        e = np.dot(X, self.weights) + self.bias  # X*W + b
        # 通过激活函数，得到全连接层的输出 o (activation_output)
        self.output = self.activate(e)
        # print(X.shape, self.weights.shape, e.shape)
        return self.output

    def activate(self, e):  # 计算激活函数的输出
        if self.activation_fun is None:
            return e  # 无激活函数，直接返回
        elif self.activation_fun == 'relu':
            return np.maximum(e, 0)
        elif self.activation_fun == 'tanh':
            return np.tanh(e)
        elif self.activation_fun == 'sigmoid':
            return np.exp(e)/ (1 + np.exp(e))
        elif self.activation_fun == 'softmax':
            for i in range(len(e)):
                tmp=e[i]
                n_max=np.max(tmp)
                tmp-=n_max
                e[i]=np.exp(tmp)/np.sum(np.exp(tmp))
            return e
        return e

    def activation_fun_derivative(self, output):
        # 计算激活函数的导数
        # 无激活函数， 导数为 1
        if self.activation_fun is None:
            return np.ones_like(output)
        # ReLU 函数的导数
        elif self.activation_fun == 'relu':
            grad = np.array(output, copy=True)
            grad[output > 0] = 1.
            grad[output <= 0] = 0.
            return grad
        # tanh 函数的导数实现
        elif self.activation_fun == 'tanh':
            return 1 - output ** 2
        # Sigmoid 函数的导数实现
        elif self.activation_fun == 'sigmoid':
            return output * (1 - output)
        elif self.activation_fun == 'softmax':
            return np.ones_like(output)

        return output


class NeuralNetwork:
    def __init__(self):
        self.layers = []  # 网络层对象列表

    def add_layer(self, layer):
        self.layers.append(layer)

    def feed_forward(self, X):
        # 前向传播（求导）
        for layer in self.layers:
            X = layer.compute(X)
        return X

    def backpropagation(self, X, y, learning_rate,bath_size,Loss_type,Regularization_type,alpha,beta):
        # 反向传播算法实现
        # 向前计算，得到最终输出值
        output = self.feed_forward(X)
        for i in reversed(range(len(self.layers))):  # 反向循环
            layer = self.layers[i]
            if layer == self.layers[-1]:  # 如果是输出层
                # 计算最后一层的 delta，(y-y`)*f`   最后一层是sigmod
                if Loss_type=='MSE':
                    layer.delta_cur = (y - output)*layer.activation_fun_derivative(layer.output)
                elif Loss_type=='MAE':
                    layer.delta_cur = np.sign(y - output) * layer.activation_fun_derivative(layer.output)
            else:  # 如果是隐藏层
                next_layer = self.layers[i + 1]
                #layer.delta变为行向量
                layer.delta_cur =np.dot( next_layer.delta_cur,next_layer.weights.T) * layer.activation_fun_derivative(layer.output)
        # 循环更新权值
        for i in range(len(self.layers)):
            layer = self.layers[i]
            # o_i 为上一网络层的输出
            o_i = X if i == 0 else self.layers[i - 1].output
            # 梯度下降算法，delta 是公式中的负数，故这里用加号
            # layer.weights += layer.delta * o_i.T * learning_rate
            # alpha=0.001 #0.001
            #reg_delta
            if Regularization_type=='l1':
                t=np.sign(layer.weights)
                layer.weights=layer.weights-t*alpha*learning_rate
            elif Regularization_type=='l2':
                layer.weights=layer.weights-learning_rate*alpha*layer.weights
            else:
                layer.weights=layer.weights
            #delta
            layer.weights += (np.dot(o_i.T,(beta*layer.delta_pre+(1-beta)*layer.delta_cur))*learning_rate)/bath_size   #10 is batch_size
            layer.bias+=np.mean((beta*layer.delta_pre+(1-beta)*layer.delta_cur),axis=0) * learning_rate
            # print('layer ',i,'\n w ',layer.weights.shape,'\n b',layer.bias.shape)
        # 更新旧梯度
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer.delta_pre=layer.delta_cur

    def train(self, X_train, X_test, y_train, y_test, learning_rate, max_epochs,bath_size,Loss_type,Rate_type,Regularization_type,alpha,beta):
        # 网络训练函数
        test_loss=[]
        test_smape=[]
        eva_loss = []
        eva_smape=[]
        train_loss = []
        train_smape=[]
        # 数据归一化
        scaler_train_X  = MinMaxScaler(feature_range=(0, 1))
        scaler_train_y = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler_train_X.fit_transform(X_train)
        y_train = scaler_train_y.fit_transform(y_train)
        scaler_test_X = MinMaxScaler(feature_range=(0, 1))
        scaler_test_y = MinMaxScaler(feature_range=(0, 1))
        X_test=scaler_test_X.fit_transform(X_test)
        y_test=scaler_test_y.fit_transform(y_test)

        for i in range(max_epochs):  # 训练 1000 个 epoch
            #learning_rate
            lr = self.learn_rate(i,Rate_type, learning_rate)
            eva_loss_tmp=[]
            eva_smape_tmp=[]
            for j in range(len(X_train)//bath_size):  # 一次训练bath_size个样本
                start=j*bath_size
                #validation
                if (start + bath_size) % 2000 == 0:
                    v_smape, v_loss = self.getloss_smape(Regularization_type, Loss_type, scaler_train_y, X_train[start:start+bath_size],
                                                               y_train[start:start+bath_size], alpha)
                    eva_loss_tmp.append(v_loss)
                    eva_smape_tmp.append(v_smape)

                self.backpropagation(X_train[start:start + bath_size], y_train[start:start + bath_size], lr, bath_size,
                                     Loss_type, Regularization_type, alpha ,beta)

            #validation
            eva_loss.append(np.mean(eva_loss_tmp))
            eva_smape.append(np.mean(eva_smape_tmp))

            #train
            t_smape, t_loss = self.getloss_smape(Regularization_type, Loss_type, scaler_train_y,X_train,y_train, alpha)
            train_loss.append(t_loss)
            train_smape.append(t_smape)
            #test
            if i % 10 == 0 or i==max_epochs-1:
                te_smape,te_loss=self.getloss_smape(Regularization_type,Loss_type,scaler_test_y,X_test,y_test,alpha)
                print('Epoch: #%s, test LOSS: %f, SMAPE: %.2f' %(i, float(te_loss),te_smape))
                test_loss .append(te_loss)
                test_smape.append(te_smape)
        return np.mean(train_loss),np.mean(train_smape),np.mean(eva_loss),np.mean(eva_smape),np.mean(test_loss),np.mean(test_smape)

    def sMAPE(self,y_true, y_pred):
        return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred + y_true))) * 100
    def reg_loss(self,Regularization_type):
        reg_loss = 0
        if Regularization_type == 'l1':
            for k in range(len(self.layers)):
                layer = self.layers[k]
                reg_loss += np.sum(np.linalg.norm(layer.weights, axis=0, ord=1))
        elif Regularization_type == 'l2':
            for k in range(len(self.layers)):
                layer = self.layers[k]
                reg_loss += np.sum(np.power(np.linalg.norm(layer.weights, axis=0, ord=2), 2))
        return reg_loss

    def learn_rate(self,i,Rate_type,learning_rate):
        lr = learning_rate
        if Rate_type == 'exponent':  # 固定下降:
            if i % 5 == 0:
                lr *= 0.96
        elif Rate_type == 'preheat':  # 学习率预热:
            if i < 20:
                lr = (learning_rate * (i + 1)) / 20
            else:
                if i % 5 == 0:
                    lr *= 0.96
        else:
            lr = learning_rate  # fixed
        return lr

    def getloss_smape(self,Regularization_type,Loss_type,scaler,X,y,alpha):
        reg_loss = self.reg_loss(Regularization_type)
        # loss_fun
        if Loss_type == 'MSE':
            loss = np.mean(np.square(y - self.feed_forward(X))) + reg_loss * alpha
        elif Loss_type == 'MAE':
            loss = np.mean(abs(y - self.feed_forward(X))) +reg_loss * alpha

        y_hat = self.feed_forward(X)
        y_hat = scaler.inverse_transform(y_hat)
        y=scaler.inverse_transform(y)
        smape = self.sMAPE(y, y_hat)
        return smape,loss
