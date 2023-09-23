import numpy as np
import pandas as pd
import argparse
from binary_classifier_dense_2layers import *


# Binary Classification

# MODEL STRUCTURE: 

#     INPUT LAYER -> 1ST HIDDEN LAYER -> 2ND HIDDEN LAYER -> OUTPUT LAYER
#     x neurons      l1 neurons           l1 neurons           1 neuron


# # # # # # # # # #  COMMAND LINE ARGUMENT PARSING  # # # # # # # # # # # # #

parser = argparse.ArgumentParser(description='Example script with command-line arguments')

parser.add_argument('--input', type = int, default = 13, help = 'Value for input variable')
parser.add_argument('--l1', type = int, default = 6, help = 'Value for neurons in first hidden layer')
parser.add_argument('--l2', type = int, default = 3, help = 'Value for neurons in second hidden layer')
parser.add_argument('--train', type = int, default = 252, help = 'Value for train variable')
parser.add_argument('--test', type = int, default = 51, help = 'Value for test variable')
parser.add_argument('--dataset', type = str, default = 'dataset.csv', help = 'Dataset name')
parser.add_argument('--print', type = int, default = 2, help = 'Value for print adjustment')
parser.add_argument('--leakrate', type = float, default = 0.01, help = 'Value for Leaky ReLU activation coefficient')         # hyperarameters of our model
parser.add_argument('--iterations', type = int, default = 10000, help = 'Value number of iterations')                         # hyperparameters of our model
parser.add_argument('--lr', type = float, default = 0.005, help = 'Value for learning rate')                                  # hyperparameters ofp our model
parser.add_argument('--threshold', type = float, default = 0.5, help = 'Value for binary threshold')                          # hyperparameters ofp our model

        # Accuracy at different Thresholds (Tuned using extensive trial and error)
        # 0.85 --> 72%
        # 0.83 --> 76%
        # 0.77 --> 78%
        # 0.73 --> 82.35%
        # 0.50 --> 94.11% 
        # 0.52 --> 94.11%
        # 0.48 --> 92%

args = parser.parse_args()

threshold = args.threshold
input_features = args.input
dataset = args.dataset
train_examples = args.train
test_examples = args.test
print_mode = args.print
iterations = args.iterations
leakrate = args.leakrate
learning_rate = args.lr
l1 = args.l1
l2 = args.l2

print(f'input_value: {input_features}')
print(f'train_value: {train_examples}')
print(f'test_value: {test_examples}')
print(f'neurons in 1st hidden layer: {l1}')
print(f'neurons in 2nd hidden layer: {l2}')
print("HYPERPARAMETERS: ")
print(f'leak rate: {leakrate}')
print(f'learning rate: {learning_rate}')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #




# # # # # # # # # #  LOADING DATASET  # # # # # # # # # # # # #

print("Reading Dataset...")

dataa = pd.read_csv(dataset)
 
# Drop the missing values
dataa = dataa.dropna()

data = dataa.copy()
  
# Apply Normalization 
for column in data.columns:
    data[column] = data[column]  / data[column].abs().max()


train = data.iloc[0:train_examples]
test = data.iloc[train_examples : train_examples + test_examples]
 
print(data.shape)
# training dataset and labels
train_input = train.iloc[:, : input_features].to_numpy().reshape(train_examples, input_features)
train_output  = train.iloc[:, input_features].to_numpy().reshape(train_examples, 1)
 

# test dataset and labels
test_input = test.iloc[:, :input_features].to_numpy().reshape(test_examples, input_features)
test_output  = test.iloc[:, input_features].to_numpy().reshape(test_examples, 1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



#  #  #  MODEL TRAINING  #  #  #

model = binary_classifier_dense_2layers(input_features, leakrate, threshold, l1, l2)
model.train(train_input, train_output, iterations, learning_rate, print_mode)
model.predict(test_input, test_output)


#  #  #        END       #  #  #
