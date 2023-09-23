import numpy as np
import matplotlib.pyplot as plt


# This is where the model lives! We used this same practice in Assignment 1 of keeping features generic and keeping the model code separate.
# THIS MODEL HAS GENERIC INPUTS FOR NEURONS IN BOTH LAYERS.



def initialize_matrices(input, neurons):
    """
    This function creates a vector of random numbers between 0, 1 of shape (input, neurons) for w and (1, neurons) for b.
    This function is basically being used to create weight and bias matrices.
    
    Argument:
    input -- how many inputs of this layer,
    neurons -- how many neurons in the layer,

    For example, a layer has 6 neurons and 13 inputs then its w vector has 6*13 attributes (13, 6) and b has 6 attributes (6 bias values for the layer, 1 for each neuron) (1, 6)
    
    Returns:
    w -- initialized vector of shape (input, neurons),
    b -- initialized vector (1, neurons)
    """
    
    w = np.random.rand(input, neurons)
    b = np.random.rand(1, neurons)

    assert(w.shape == (input, neurons))
    assert(b.shape == (1, neurons))
    
    return w, b



class binary_classifier_dense_2layers:
    """
    Model for a dense neural network of 2 hidden layers (6 and 3 neurons respectively)
    """

    def __init__(self, input_features, leakrate, threshold, l1, l2):
        """
        This function is the constructor of the model/class whatever you want to call it!
        It initializes the weight matrics and model descriptors.
    
        Argument:
        input_features -- This specified how many input neurons are present in input layer
        l1 -- number of neurons in firsts hidden layer
        l2 -- number of neurons in second hiddel layer

        Author's Note: This has been thoroughly tested on the dataset we're working on. Firstly, we hardcoded the input_features value and then made it generic. It may cause bugs on other datasets though it should not.
    
        """

        self.parameters = []                                    # Parameters of the model. parameters[0] are the parameters of 1st hidden layer, where in parameters[0][0] is weight matrix of first hidden layer and parameters[0][1] is its bias matrix. 
        self.layers = 2                                         # No. of hidden layers
        self.neurons = [input_features, l1, l2, 1]              # Model Description
        self.leak_rate = leakrate                               # Leak rate for Leaky ReLU Activation Function
        self.threshold = threshold                              # Threshold for Binary Classification

        # Parameter data structure
        for i in range(self.layers + 1):

            w, b = initialize_matrices(self.neurons[i], self.neurons[i + 1])

            array = []
            array.append(w)     # [0] -> weights
            array.append(b)     # [1] -> bias

            self.parameters.append(array)


    # Forward Propagation Module
    def forward_propagation(self, train_input):
        """
        This module performs forward propagation on train_input (or you can call it any data you pass.)
    
        Argument:
        train_input -- Data to perform FP on. THIS IS NOT limited to training data! This name was used in early stages of writing of this code and wasn't changed later to avoid bugs.

        Returns:
        output_matrix -- The activated values of output neuron,
        activations -- activated values of each neuron and layers for the propagation (to be used in back propagation for computing derivates),
        prediction -- actual predictions (0 or 1).. scaled using a threshold
    
        """
        
        output_matrix = train_input
        activations = []

        for i in range(self.layers):

            aux = np.dot(output_matrix, self.parameters[i][0]) + self.parameters[i][1]
            output_matrix = np.maximum(self.leak_rate * aux, aux)                                                                   # using Leaky ReLU activation function on hidden layers with leak factor of 0.01
            activations.append(output_matrix)

            assert(output_matrix.shape == (train_input.shape[0], self.neurons[i + 1]))

            # output_matrix is the thing to pass to the next layer.

        # Now pass output_matrix to output neuron

        output_matrix = np.dot(output_matrix, self.parameters[self.layers][0]) + self.parameters[self.layers][1] 
        output_matrix = 1 / (1 + np.exp(-output_matrix))                                                                           # using sigmoid activation at output layer for obvious reasons (binary classification)

        predictions = (output_matrix >= self.threshold) * 1.0

        # Accuracy at different Thresholds (Tuned using extensive trial and error)
        # 0.85 --> 72%
        # 0.83 --> 76%
        # 0.77 --> 78%
        # 0.73 --> 82.35%
        # 0.50 --> 94.11% 
        # 0.52 --> 94.11%
        # 0.48 --> 92%

    
        assert(predictions.shape == (train_input.shape[0], 1))

        return output_matrix, predictions, activations
    

    # Cost Function Computation Module
    def costFunction(self, predictions, train_output): 
        """
        This function computes the cost function.
    
        Argument:
        predictions -- activated outputs of output neuron,
        train_output -- true labels

        Returns:
        loss -- Binary Cross Entropy Loss
    
        """

        logprobs = np.multiply(np.log(predictions), train_output) + np.multiply((1 - train_output), np.log(1 - predictions))
        loss = - np.sum(logprobs) / train_output.shape[0]

        return loss
    
    # Back Propagation Module
    def backward_propagation(self, activations, train_input, train_output, predictions):
        """
        This module performs back propagation. It propagates the error backwards and computes derivates on error surface with respect to weights and biases.
    
        Argument:
        activations  -- Activated values of each neuron produced by this iteration's forward propagation [A],
        train_input  -- Training Data [X],
        train_output -- True Labels [Y],
        predictions  -- Activated Output Sigmoid Values [Y'],

        Returns:
        dw -- A matrix of SAME dimensions as the self.parameters's weight matrices but with values of dw of each weight of the model,
        db -- A matrix of SAME dimensions as the self.parameters's weight matrices but with values of db of each bias of the model.
    
        """
        dw = []
        db = []

        # First Compute the derivates for output layer differently as we're using sigmoid activation here. 

        dwo = (np.dot(activations[1].T, (predictions - train_output)))/train_input.shape[0]     # dwo = (A.T * (Y' - Y))/ m  ; where m is number of training examples
        assert(dwo.shape == (self.neurons[2], 1))
        dbo = np.mean(predictions - train_output)                                               # dbo = (1 * (Y' - Y))/ m    ; where m is number of training examples

        dw.append(dwo)
        db.append(dbo)

        i = 2                       # 2 layers
        pass_back = []              # values to pass back to previous layer for chain rule enforcement

        while i > 0:

            i -= 1

            dwh = []                # DW matrix for this LAYER.
            dbh = []

            for j in range(self.neurons[i + 1]):        # i'th layer has self.neurons[i + 1] number of neurons

                # Compute derivates for each neuron differently of this layer and combine them into a matrix.
                dwn = []            # for particular neuron of a particular layer
                dbn = []

                if i > 0:           # IF this is not the 1st hidden layer (i.e. not directly connected to input layer)

                    negative_indices = activations[i][:, j] < 0

                    A = (predictions - train_output)*self.parameters[i + 1][0][j, 0]
                    A[negative_indices] *= self.leak_rate
                    pass_back.append(A)
                    dbn = np.mean(A)

                    dwn = np.dot(activations[i - 1].T, A)/train_input.shape[0]
                    assert(dwn.shape == (self.neurons[i], 1))
                    dwh.append(dwn)

                    # Compute BIAS
                    dbh.append(dbn)


                else:                # IF this is the 1st hidden layer (i.e. directly connected to input layer) Here, instead of using activations, we use train_input and combine errors from all nodes of next layer.
                
                    # Add up passed back errors.
                    
                    A = []

                    for m in range(self.neurons[2]):
                        if m == 0:
                            A = pass_back[m]*self.parameters[i + 1][0][j, m]
                        else:
                            A += pass_back[m]*self.parameters[i + 1][0][j, m]

                    negative_indices = activations[i][:, j] < 0
                    A[negative_indices] *= self.leak_rate                       # Leaky ReLU differential is 1 for positive and 0.01 for negative.

                    dbn = np.mean(A)
                    dbh.append(dbn)

                    dwn = np.dot(train_input.T, A)/train_input.shape[0]         # train_input.T is of size = (input_features, m) and D is of size(m, 1) so their dot gives matrix of size (input_features, 1) i.e. WEIGHTS FOR THIS NEURON!
                    assert(dwn.shape == (self.neurons[i], 1))
                    dwh.append(dwn)
            

            # Put together all the dwh in dwn
            stack1 = dwh[0]
            stack2 = dbh[0]

            m = 1

            while m < self.neurons[i + 1]:
                stack1 = np.hstack((stack1, dwh[m]))
                stack2 = np.hstack((stack2, dbh[m]))
                m += 1

            dw.append(stack1)
            db.append(stack2)
            
        # we also need to reverse dw and db for effective subtraction.
        dw = dw[::-1]
        db = db[::-1]

        return dw, db
    
    def update(self, dw, db, learning_rate):
        """
        This module updates the weights globally once back propagation is complete.
    
        Argument:
        dw            -- weight derivates of the model,
        db            -- bias derivates of the model,
        learning_rate -- learning rate (speed of traversal),
    
        """

        self.parameters[0][0] = self.parameters[0][0] - learning_rate*dw[0]         # update 1st hidden layer weights
        self.parameters[0][1] = self.parameters[0][1] - learning_rate*db[0]

        self.parameters[1][0] = self.parameters[1][0] - learning_rate*dw[1]         # update 2nd hidden layer weights
        self.parameters[1][1] = self.parameters[1][1] - learning_rate*db[1]

        self.parameters[2][0] = self.parameters[2][0] - learning_rate*dw[2]         # update output layer weights
        self.parameters[2][1] = self.parameters[2][1] - learning_rate*db[2]

    
    def train(self, train_input, train_output, iterations = 1000, learning_rate = 0.05, print_mode = -1):
        """
        This module combines other training modules in sequence for effective training.
    
        Argument:
        train-input     -- train data vector,
        train_output    -- true labels vector,
        iterations      -- number of iterations
        learning_rate   -- learning rate (speed of traversal),
        print_mode      -- for controlled printing of cost. Default value is -1

        """
        offset = 1
        accuracy_rec = []
        iterations_rec = []
        cost_rec = []

        print()

        # Adjust cost printing
        if print_mode == -1:
            if iterations > 50:
                offset = iterations / 20

        else:
            if print_mode == 2 and iterations > 19:
                offset = iterations / 20

        for i  in range(iterations):

            output_matrix, predictions, activations = self.forward_propagation(train_input)
            cost = self.costFunction(output_matrix, train_output)

            accuracy = np.mean(predictions == train_output)*100

            # For graphing,
            cost_rec.append(cost)
            iterations_rec.append(i + 1)
            accuracy_rec.append(accuracy)
            
            if i % offset == 1:
                print("COST AFTER ITERATION NO. {} is {} and accuracy = {}".format(i+1, cost, accuracy))

            dw, db = self.backward_propagation(activations, train_input, train_output, output_matrix)
            self.update(dw, db, learning_rate)

        # Model is trained!
        print()
        print("Model is trained!")

        # PRINT GRAPHS

        plt.plot(iterations_rec, cost_rec, label = 'Cost values at iterations')
        plt.xlabel('Iterations')
        plt.ylabel('COST')
        plt.legend()
        plt.show()

        plt.plot(iterations_rec, accuracy_rec, label = 'Accuracy values at iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


    def predict(self, test_input, test_output):
        """
        This module peforms FP on test data.
    
        Argument:
        test_input     -- test data vector,
        test_output    -- true test labels vector,

        """

        om, predictions, activations = self.forward_propagation(test_input)
        print()
        print()
        print("TEST SET PREDICTIONS ARE:    ")
        print()
        print(predictions)
        print()
        print()

        results = np.mean(predictions == test_output)*100
        print("Test Set Accuracy = {}".format(results))
        print()
        print("-------------------------------------------------------")

    def computeVal(self, test_case):
        """
        This module (AFTER LEARNING!) predicts a 0 or 1 for given test case.
    
        Argument:
        test_case -- Test case to perform binary classification on.

        """
        om, predictions, activations = self.forward_propagation(test_case)
        return predictions

