# Binary Classifier (Dense Neural Network -- 2 Hidden Layers)
This is a dense, deep neural network consisting of 2 hidden layers capable of binary classification.



				                            DEEP, DENSE NEURAL NETWORK
			                                    Authors: Mukul Malik



## Contents
1) How to Use the Model and Tune Hyperparameters
2) Description and Model Architecture
3) Activation Functions

For testing, 2 different datasets are also available.

5) Dataset1 Description
6) DATASET2 Description

## HOW TO TRAIN AND TUNE HYPERPARAMETERS:

binary_classifier_dense_2layers.py file contains the model code and train.py has code to invoke training.

To train on the dataset we have supplied (`dataset.csv` -- it's description is given at the end),
simply run: 

`python train.py`

as we have set the default parameters to work for this dataset. All tuned hyperparameters are already 
set for `dataset.csv`. These were determined after extensive trial and error.

The model has 3 hyperparameters: learning rate, leak rate (leaky reLU coefficient) and threshold for
binary classification. These can be changed as per your wish in ways described below.

`train.py` takes 11 optional arguments: (for changing default settings, using other datasets etc.)

1) `--input`      : An integer to pass the number of input features in the dataset

2) `--train`      : To pass the number of rows of your dataset to use for training

3) `--test `      : To pass the number of rows of your dataset to use for testing

4) `--dataset`    : To pass the name of your dataset

5) `--lr`         : learning rate

6) `--leakrate`   : leak rate coefficient for Leaky ReLU activation

7) `--iterations` : number of iterations to train for

8) `--threshold`  : threshold for binary classification

9) `--print`      : takes a value of 1 or 2. 1 specifies that you wish to print cost and accuracy after
		  each iteration. Iterations above 1000 will cause your screen to be covered with too
		  much text. In such a case you can use print value '2' which will scale printing to 
		  after a certain offset of iterations to not print more than 20 values (spread evenly
 		  between 0 and max iterations.)

10) `--l1` 	: number of neurons in first hidden layer [default set to 6]

11) `--l2` 	: number of neurons in second hidden layer [default set to 3]

NOTE: We recommend not changing default leakrate and threshold (set at 0.01 and 0.5 respectively).
They're perfectly tuned for most variety of data. If you don't supply a print value, it will
automatically determine the best printing mode to avoid cluttering of screen with useless text.


For example, suppose your dataset name is `'dataset2.csv'` and it has 14 input features and you want to use first 252 rows for train and rest 51 rows for test. With 10000 iterations and learning rate of 0.05 and you want 36 neurons in first layer and 16 neurons in second, then you invoke:

`python train.py --dataset 'Datasets/dataset2.csv' --input 14 --train 252 --test 51 --iterations 10000 --lr 0.05 --l1 36 --l2 16`

## Description and Model Architecture

2 Hidden layers. Some good rules for choosing number of neurons [l1 and l2 parameter]

1. For most problems, one could probably get decent performance (even without a second optimization step) 
   by setting the hidden layer configuration using just two rules: (i) the number of hidden layers 
   equals one; and (ii) the number of neurons in that layer is the mean of the neurons in the input 
   and output layers. 
   [Source: [https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw](url)]

2. We can usually prevent over-fitting if we keep number of neurons below:
   Nh = Ns / (α *(Ni + No))

   Ni : number of input neurons.
   No : number of output neurons
   Ns : number of samples in training data set.
   α  : an arbitrary scaling factor usually 2-10. [usually kept at 5]
   [Source: [https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw](url)]

## ACTIVATION FUNCTION (Leaky ReLU + Sigmoid)

We decided to use Leaky ReLU Activation Function for hidden layers. 

Since this is a binary classification model, sigmoid activation was used at output neuron (layer).


## DATASET 1 DESCRIPTION (dataset.csv)

The first dataset has 13 input features: 

`Age` :           Age of the patient

`Sex `: 		Sex of the patient

`exang`: 		exercise induced angina (1 = yes; 0 = no)

`ca`: 		number of major vessels (0-3)

`cp` : 		Chest Pain type chest pain type
     			Value 1: typical angina
     			Value 2: atypical angina
     			Value 3: non-anginal pain
     			Value 4: asymptomatic

`trtbps` : 	resting blood pressure (in mm Hg)

`chol` : 		cholestoral in mg/dl fetched via BMI sensor

`fbs` : 		(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

`rest_ecg` : 	resting electrocardiographic results
         		Value 0: normal
         		Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
         		Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

`thalach` : 	maximum heart rate achieved

`target` : 	0= less chance of heart attack 1= more chance of heart attack	[TRUE LABEL]

[Acquired from [https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset](url)]


## DATASET2 DESCRIPTION

This dataset has 14 input features: 

`Workclass` :           	Workclass of the person

`Education` : 		Education of the patient
			Value Matrix:

			EDUCATION	NUMBER
			Doctorate	1
			12th		2
			Bachelors	3
			Some-college	4
			7th-8th		5
			9th		6
			HS-grad		7
			10th		8
			11th		9
			Masters		10
			Preschool	11
			Prof-school	12
			5th-6th		13
			1st-4th		14
			Assoc-acdm	15


`fnlwgt`: 		net worth

`educational-num`: 	educatinal qualification the person

`marital-status` : 	Chest Pain type chest pain type

			Value Matrix:

			Marital-status		NUMBER
			Divorced		1
			Never-married		2
			Married-civ-spouse	3
			Widowed			4
			Married-spouse-absent	5

     			

`occupation` : 		occupation of the person

			Value Matrix:

			Occupation		NUMBER
			Exec-managerial		1
			Other-service		2
			Transport-moving	3
			Adm-clerical		4
			Machine-op-inspct	5
			Sales			6
			Handlers-cleaners	7
			Farming-fishing		8
			Protective-serv		9
			Prof-specialty		10
			Craft-repair		11
			Tech-support		12
			Protective-serv		13
			Priv-house-serv		14


`relationship` : 		relationship of the person

			Value Matrix:

			Relationship	number
			Not-in-family	1
			Own-child	2
			Husband		3
			Wife		4
			Unmarried	5
			Other-relative	6


`race` : 			race of the person

			Value Matrix:

			Race			NUMBER
			White			1
			Black			2
			Asian-Pac-Islander	3
			Amer-Indian-Eskimo	4

`gender` : 		gender of the person

			Value Matrix:

			Gender	NUMBER
			Male	1
			Female	2

         		

`capital_gain` : 		capital_gain of the person

`capital_loss` : 		capital_loss of the person

`working_hours`:		working hours of the person per week

`native_country`:		country of the person

			Value Matrix:

			NATIVE-COUNTRY			Number
			United-States			1
			Japan				2
			South				3
			Portugal			4
			Italy				5
			Ecuador				6
			Mexico				7
			England				8
			Philippines			9
			China				10
			Canada				11
			Dominican-Republic		12
			Jamaica				13
			Germany				14
			Vietnam				15
			Puerto-Rico			16
			Thailand			17
			Cuba				18
			India				19
			Cambodia			20
			Yugoslavia			21
			Iran				22
			El-Salvador			23
			Poland				24
			Greece				25
			Ireland				26
			Guatemala			27
			Scotland			28
			Hong				29
			Peru				30
			Haiti				31
			Hungary				32
			France				33
			Nicaragua			34
			Laos				35
			Taiwan				36
			Outlying-US(Guam-USVI-etc)	37


`TARGET`:			0 if income will be less than 50k and 1 if more.
