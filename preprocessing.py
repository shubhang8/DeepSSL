import sys
import h5py
import numpy as np
from scipy.io import loadmat

def load_files(train_file, test_file):
    print("loading training data")
    with h5py.File(train_file, 'r') as train_data:
        train_labels = np.transpose(train_data['traindata'], (1, 0))
        train_labels = train_labels[0:1000]
        # print(train_labels.shape)

        traindata = train_data['trainxdata']
        train_inputs = np.empty((1000, 1000), dtype=np.float32)
        train_inputs = np.transpose(np.argmax(traindata[:, :, 0:1000], axis=1))
        # print(train_inputs.shape)

    print("loading testing data")
    test_data = loadmat(test_file)

    test_labels = test_data['testdata']
    test_inputs = np.argmax(test_data['testxdata'], axis = 1)

    # print(test_labels.shape)
    # print(test_inputs.shape)

    return train_inputs, train_labels, test_inputs, test_labels

def get_kmers(inputs, labels, dna_dict, len_kmer):
    kmers = np.empty((len(inputs) + 1, 3), dtype=object)
    kmers[0] = "sequence", "fake_label", "real_label"

    for i in range(1, len(inputs)+ 1):
        # just to track time in big files -- can delete this
        if i % 10000 == 0:
            print(i/len(inputs))
        kmers[i, 0] = ''
        seq = ''
        for j in range(len(inputs[0])):
            seq += dna_dict[inputs[i-1,j]] # get sequence

        num_kmers = 0
        for j in range(245, 755):
            kmers[i, 0] += seq[j:j+k]
            kmers[i, 0] += ' ' # separate kmers by spaces
            num_kmers += 1

        kmers[i, 1] = 1 # fake label
        # kmers[i, 2] = labels[i]
        kmers[i, 2] = ''
        for j in range(len(labels[0])):
            kmers[i, 2] += str(labels[i-1, j]) # real label
            kmers[i, 2] += ' '
    # print(num_kmers)
    return kmers

## MIGHT NEED TO CHANGE THESE
train_file = './deepsea_train/train.mat'
test_file = './deepsea_train/test.mat'

train_inputs, train_labels, test_inputs, test_labels = load_files(train_file, test_file)


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
np.savetxt('./DNABert/examples/DeepSea_data/3_small/dev.tsv', test_kmers, fmt='%s', delimiter='\t')

print("starting training 3-mers")
train_kmers = get_kmers(train_inputs, train_labels, dna_dict, k)
np.savetxt('./DNABert/examples/DeepSea_data/3_small/train.tsv', train_kmers, fmt='%s', delimiter='\t')

k = 4
print("k = 4")

print("starting testing 3-mers")
test_kmers = get_kmers(test_inputs, test_labels, dna_dict, k)
np.savetxt('./DNABert/examples/DeepSea_data/4_small/dev.tsv', test_kmers, fmt='%s', delimiter='\t')

print("starting training 3-mers")
train_kmers = get_kmers(train_inputs, train_labels, dna_dict, k)
np.savetxt('./DNABert/examples/DeepSea_data/4_small/train.tsv', train_kmers, fmt='%s', delimiter='\t')


k = 5
print("k = 5")

print("starting testing 3-mers")
test_kmers = get_kmers(test_inputs, test_labels, dna_dict, k)
np.savetxt('./DNABert/examples/DeepSea_data/5_small/dev.tsv', test_kmers, fmt='%s', delimiter='\t')

print("starting training 3-mers")
train_kmers = get_kmers(train_inputs, train_labels, dna_dict, k)
np.savetxt('./DNABert/examples/DeepSea_data/5_small/train.tsv', train_kmers, fmt='%s', delimiter='\t')


k = 6
print("k = 6")

print("starting testing 3-mers")
test_kmers = get_kmers(test_inputs, test_labels, dna_dict, k)
np.savetxt('./DNABert/examples/DeepSea_data/6_small/dev.tsv', test_kmers, fmt='%s', delimiter='\t')

print("starting training 3-mers")
train_kmers = get_kmers(train_inputs, train_labels, dna_dict, k)
np.savetxt('./DNABert/examples/DeepSea_data/6_small/train.tsv', train_kmers, fmt='%s', delimiter='\t')

