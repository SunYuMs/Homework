import numpy as np
from sklearn.preprocessing import MinMaxScaler
train = np.loadtxt(open("./data/train.csv","rb"),delimiter=",")
test=np.loadtxt(open("./data/test.csv","rb"),delimiter=",")
Input_length=60
Out_length=56
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(sequences[0])):

        end_element_index = i + n_steps_in
        out_end_index = end_element_index + n_steps_out

        if out_end_index > len(sequences[0]):
            break

        sequence_x, sequence_y = sequences[:,i:end_element_index], sequences[:,end_element_index:out_end_index]
        X.append(sequence_x)
        y.append(sequence_y)

    return np.array(X), np.array(y)

train_X,train_y=split_sequences(train,Input_length,Out_length)
# test.reshape(1,)
train_X=train_X.reshape(train_X.shape[0]*train_X.shape[1],train_X.shape[2])
train_y=train_y.reshape(train_y.shape[0]*train_y.shape[1],train_y.shape[2])
#68820*60