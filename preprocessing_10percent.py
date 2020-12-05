import sys
import h5py
import numpy as np
from scipy.io import loadmat
import time

def load_files(file):
    print("loading  data")
    data = loadmat(file)

    labels = data['testdata']
    inputs = np.argmax(data['testxdata'], axis = 1)

    train_inputs = inputs[0:int(0.9 * len(inputs))]
    test_inputs = inputs[int(0.9 * len(inputs)):len(inputs)]

    train_labels = labels[0:int(0.9 * len(labels))]
    test_labels = labels[int(0.9 * len(labels)):len(labels)]


    return train_inputs, train_labels, test_inputs, test_labels

def get_kmers(inputs, labels, dna_dict, len_kmer):
    kmers = np.empty((len(inputs), 3), dtype=object)

    for i in range(len(inputs)):
        # just to track time in big files -- can delete this
        if i % 10000 == 0:
            print(i/len(inputs))
        kmers[i, 0] = ''
        seq = ''
        for j in range(len(inputs[0])):
            seq += dna_dict[inputs[i,j]] # get sequence

        for j in range(len(inputs[0]) - k):
            kmers[i, 0] += seq[j:j+k]
            kmers[i, 0] += ' ' # separate kmers by spaces

        kmers[i, 1] = 1 # fake label
        # kmers[i, 2] = labels[i]
        kmers[i, 2] = ''
        for j in range(len(labels[0])):
            kmers[i, 2] += str(labels[i, j]) # real label
            kmers[i, 2] += ' '
    print(kmers[0])
    return kmers

train_inputs, train_labels, test_inputs, test_labels = load_files('test.mat')

dna_dict = {
    0: "A",
    1: "C",
    2: "T",
    3: "G"
}

k = 3
print("k = 3")

print("starting testing 3-mers")
test_kmers = get_kmers(test_inputs, test_labels, dna_dict, k)
np.savetxt('test_3.txt', test_kmers, fmt='%s', delimiter='\t')

print("starting training 3-mers")
train_kmers = get_kmers(train_inputs, train_labels, dna_dict, k)
np.savetxt('train_3.txt', train_kmers, fmt='%s', delimiter='\t')


k = 4
print("k = 4")

print("starting testing 4-mers")
test_kmers = get_kmers(test_inputs, test_labels, dna_dict, k)
np.savetxt('test_4.txt', test_kmers, fmt='%s', delimiter='\t')

print("starting training 4-mers")
train_kmers = get_kmers(train_inputs, train_labels, dna_dict, k)
np.savetxt('train_4.txt', train_kmers, fmt='%s', delimiter='\t')

k = 5
print("k = 5")

print("starting testing 5-mers")
test_kmers = get_kmers(test_inputs, test_labels, dna_dict, k)
np.savetxt('test_5.txt', test_kmers, fmt='%s', delimiter='\t')

print("starting training 5-mers")
train_kmers = get_kmers(train_inputs, train_labels, dna_dict, k)
np.savetxt('train_5.txt', train_kmers, fmt='%s', delimiter='\t')

k = 6
print("k = 6")

print("starting testing 6-mers")
test_kmers = get_kmers(test_inputs, test_labels, dna_dict, k)
np.savetxt('test_6.txt', test_kmers, fmt='%s', delimiter='\t')

print("starting training 6-mers")
train_kmers = get_kmers(train_inputs, train_labels, dna_dict, k)
np.savetxt('train_6.txt', train_kmers, fmt='%s', delimiter='\t')
