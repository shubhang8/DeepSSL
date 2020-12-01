import sys
import h5py
import numpy as np
from scipy.io import loadmat

# sys.argv should be file paths to train.mat, valid.mat, test.mat

def preprocess(train_file, valid_file, test_file):
    with h5py.File(train_file, 'r') as train_data:
        train_labels = np.transpose(train_data['traindata'], (1, 0))

        traindata = train_data['trainxdata']
        train_inputs = np.empty((4400000, 1000), dtype=np.float32)
        for i in range(int(4400000/1000)):
            train_inputs[i*1000:(i+1)*1000] = np.transpose(np.argmax(traindata[:, :, i*1000:(i+1)*1000], axis=1))


    valid_data = loadmat(valid_file)

    valid_labels = valid_data['validdata']
    valid_inputs = np.argmax(valid_data['validxdata'], axis = 1)


    test_data = loadmat(test_file)

    test_labels = test_data['testdata']
    test_inputs = np.argmax(test_data['testxdata'], axis = 1)

    return train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels

train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels = preprocess(sys.argv[1], sys.argv[2], sys.argv[3])
