import os
import numpy as np
# from preprocess import get_training_data, get_testing_data, get_next_batch
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 100
num_batches = 600
num_epochs = 10

class KmerClassifer(nn.Module):
    def __init__(self,):
        super(KmerClassifer, self).__init__()
        # self.conv1 = nn.Conv2d(768,850,(1, 8),(1, 1))
        # self.conv2 = nn.Conv2d(850,960,(1, 8),(1, 1))
        self.conv1 = nn.Conv1d(768,850,8)
        self.conv2 = nn.Conv1d(850,960,8)
        self.pool = nn.MaxPool1d(4,4)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(960,925)
        self.linear2 = nn.Linear(925,919)
        self.sig = nn.Sigmoid()
		#nn.Threshold(0, 1e-06)
		#nn.Threshold(0, 1e-06)
		#nn.MaxPool2d((1, 4),(1, 4))
		#Lambda(lambda x: x.view(x.size(0),-1)), # Reshape,
		#nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(50880,925)), # Linear,
		#nn.Threshold(0, 1e-06),
		#nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(925,919)), # Linear,


    def forward(self, inputs): # similar to call function
        print("forward")
        inputs = inputs.transpose(1, 2)
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.pool(x)
        conv1 = self.drop1(x)

        y = self.conv2(conv1)
        y = F.relu(y)
        #y = self.pool(y)
        conv2 = self.drop2(y)

        #conv2 = conv2.view(-1, 53*960)
        print("conv2 shape: ",conv2.shape)
        conv2 = conv2.transpose(1, 2)
        linear1 = self.linear1(conv2)
        #linear1 = F.relu(z)

        linear2 = self.linear2(linear1)

        sig = self.sig(linear2)

        return sig

class ClsClassifer(nn.Module):

	def __init__(self,):
		"""
		The model class inherits from nn.Module.
		It stores the trainable weights as class members.
		"""
		super(ClsClassifer, self).__init__()
		self.linear = nn.Linear(768,919)
		self.sig = nn.Sigmoid()

	def forward(self, inputs):
		x = self.linear(inputs)
		x = self.sig(x)

		return x

# def train(model, train_input, train_labels):
# 	"""
# 	Runs through one epoch - all training examples
# 	"""
# 	loss = nn.CrossEntropyLoss()
# 	for j in range(num_batches):
#         imgs, anss = get_next_batch(j, train_input, train_labels)
#         imgs = torch.tensor(imgs) # turns it into torch tensor, cannot directly pass in numpy arrays

#     logits = model(imgs) # not probabilities!
# 	l = loss(logits, torch.tensor(anss))
# 	model.optimizer.zero_grad()
# 	l.backward()
# 	model.optimizer.step() # can be thought of as gradient descent

# def test(model, test_input, test_labels):
# 	"""
# 	Runs through all test examples

# 	:returns: sum of each batch's accuracy
# 	"""
# 	sum_acc = 0
# 	for i in range(100):
# 		imgs, anss = get_next_batch(i, test_input, test_labels)
# 		imgs = torch.tensor(imgs)
# 		logits = model(imgs).detach().numpy() # must call detach
# 		sum_acc += model.accuracy_function(logits, anss)
# 	return sum_acc

# def main():
# 	train_input, train_labels = get_training_data()
# 	test_input, test_labels = get_testing_data()
# 	model = Model()
# 	for i in range(num_epochs):
# 		train(model, train_input, train_labels)
# 		print("Epoch: ", i)
# 		sum_acc = test(model, test_input, test_labels)
# 		print("Test Accuracy: %r" % (sum_acc/100))
# 	return

# # if __name__ == '__main__':
# # 	main()
