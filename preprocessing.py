import sys
import h5py
import numpy as np
from scipy.io import loadmat

def load_files(train_file, valid_file, test_file):
    print("loading training data")
    with h5py.File(train_file, 'r') as train_data:
        train_labels = np.transpose(train_data['traindata'], (1, 0))

        traindata = train_data['trainxdata']
        train_inputs = np.empty((4400000, 1000), dtype=np.float32)
        for i in range(int(4400000/1000)):
            train_inputs[i*1000:(i+1)*1000] = np.transpose(np.argmax(traindata[:, :, i*1000:(i+1)*1000], axis=1))

    print("loading validation data")
    valid_data = loadmat(valid_file)

    valid_labels = valid_data['validdata']
    valid_inputs = np.argmax(valid_data['validxdata'], axis = 1)

    print("loading testing data")
    test_data = loadmat(test_file)

    test_labels = test_data['testdata']
    test_inputs = np.argmax(test_data['testxdata'], axis = 1)
    return train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels

def get_kmers(inputs, labels, dna_dict, len_kmer):
    kmers = np.empty((len(inputs), 3), dtype=object)

    for i in range(len(inputs)):
        # just to track time in big files -- can delete this
        if i % 10000 == 0:
            print(i/len(inputs))

        curr_index = 0
        kmers[i, 0] = '' #kmers separated by spaces
        while curr_index < len(inputs[0]):
            for k in range(len_kmer):
                kmers[i, 0] += dna_dict[inputs[i, curr_index]]
                curr_index += 1
            kmers[i, 0] += " "
        kmers[i, 1] = 1 # fake label
        kmers[i, 2] = labels[i] # real label
    return kmers

k = 4
dna_dict = {
    0: "A",
    1: "C",
    2: "T",
    3: "G"
}

train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels = load_files('train.mat', 'valid.mat', 'test.mat')

print("starting validation kmers")
valid_kmers = get_kmers(valid_inputs, valid_labels, dna_dict, k)
np.savetxt('valid_4.gz', valid_kmers, fmt='%s', delimiter='\t')

print("starting testint kmers")
test_kmers = get_kmers(test_inputs, test_labels, dna_dict, k)
np.savetxt('test_4.gz', test_kmers, fmt='%s', delimiter='\t')

print("starting training kmers")
train_kmers = get_kmers(train_inputs, train_labels, dna_dict, k)
np.savetxt('train_4.gz', train_kmers, fmt='%s', delimiter='\t')
