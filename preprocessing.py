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
        # # just to track time in big files -- can delete this
        # if i % 10000 == 0:
        #     print(i/len(inputs))
        kmers[i, 0] = ''
        seq = ''
        for j in range(len(inputs[0])):
            seq += dna_dict[inputs[i,j]] # get sequence

        for j in range(len(inputs[0]) - k):
            kmers[i, 0] += seq[j:j+k]
            kmers[i, 0] += ' ' # separate kmers by spaces

        kmers[i, 1] = 1 # fake label
        kmers[i, 2] = labels[i] # real label

    return kmers

## MIGHT NEED TO CHANGE THESE
train_file = 'train.mat'
valid_file = 'valid.mat'
test_file = 'test.mat'

train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels = load_files(train_file, valid_file, test_file)


dna_dict = {
    0: "A",
    1: "C",
    2: "T",
    3: "G"
}

k = 3
print("k = 3")

print("starting validation 3-mers")
valid_kmers = get_kmers(valid_inputs, valid_labels, dna_dict, k)
np.savetxt('valid_3.gz', valid_kmers, fmt='%s', delimiter='\t')

print("starting testing 3-mers")
test_kmers = get_kmers(test_inputs, test_labels, dna_dict, k)
np.savetxt('test_3.gz', test_kmers, fmt='%s', delimiter='\t')

print("starting training 3-mers")
train_kmers = get_kmers(train_inputs, train_labels, dna_dict, k)
np.savetxt('train_3.gz', train_kmers, fmt='%s', delimiter='\t')

k = 4
print("k = 4")

print("starting validation 4-mers")
valid_kmers = get_kmers(valid_inputs, valid_labels, dna_dict, k)
np.savetxt('valid_4.gz', valid_kmers, fmt='%s', delimiter='\t')

print("starting testing 4-mers")
test_kmers = get_kmers(test_inputs, test_labels, dna_dict, k)
np.savetxt('test_4.gz', test_kmers, fmt='%s', delimiter='\t')

print("starting training 4-mers")
train_kmers = get_kmers(train_inputs, train_labels, dna_dict, k)
np.savetxt('train_4.gz', train_kmers, fmt='%s', delimiter='\t')k = 3

k = 5
print("k = 5")

print("starting validation 5-mers")
valid_kmers = get_kmers(valid_inputs, valid_labels, dna_dict, k)
np.savetxt('valid_5.gz', valid_kmers, fmt='%s', delimiter='\t')

print("starting testing 5-mers")
test_kmers = get_kmers(test_inputs, test_labels, dna_dict, k)
np.savetxt('test_5.gz', test_kmers, fmt='%s', delimiter='\t')

print("starting training 5-mers")
train_kmers = get_kmers(train_inputs, train_labels, dna_dict, k)
np.savetxt('train_5.gz', train_kmers, fmt='%s', delimiter='\t')k = 3

k = 6
print("k = 6")

print("starting validation 6-mers")
valid_kmers = get_kmers(valid_inputs, valid_labels, dna_dict, k)
np.savetxt('valid_6.gz', valid_kmers, fmt='%s', delimiter='\t')

print("starting testing 6-mers")
test_kmers = get_kmers(test_inputs, test_labels, dna_dict, k)
np.savetxt('test_6.gz', test_kmers, fmt='%s', delimiter='\t')

print("starting training 6-mers")
train_kmers = get_kmers(train_inputs, train_labels, dna_dict, k)
np.savetxt('train_6.gz', train_kmers, fmt='%s', delimiter='\t')
