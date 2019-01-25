import numpy as np
from sklearn.preprocessing import OneHotEncoder


class NeuralNetwork:
    """
    A multi-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices.

    The network uses a nonlinearity after each fully connected layer except for the
    last. You will implement two different non-linearities and try them out: Relu
    and sigmoid.

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_sizes, output_size, num_layers, nonlinearity='relu'):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H_1)
        b1: First layer biases; has shape (H_1,)
        .
        .
        Wk: k-th layer weights; has shape (H_{k-1}, C)
        bk: k-th layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: List [H1,..., Hk] with the number of neurons Hi in the hidden layer i.
        - output_size: The number of classes C.
        - num_layers: Number of fully connected layers in the neural network.
        - nonlinearity: Either relu or sigmoid
        """
        self.num_layers = num_layers
       

        assert(len(hidden_sizes)==(num_layers-1))
        sizes = [input_size] + hidden_sizes + [output_size]
        

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params['W' + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params['b' + str(i)] = np.zeros(sizes[i])
        
        
            
        if nonlinearity == 'sigmoid':
            self.nonlinear = self.sigmoid
            self.nonlinear_grad = self.sigmoid_grad
        elif nonlinearity == 'relu':
            self.nonlinear = self.relu
            self.nonlinear_grad = self.relu_grad
        

    def forward(self, X):
        """
        Compute the scores for each class for all of the data samples.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.

        Returns:
        - scores: Matrix of shape (N, C) where scores[i, c] is the score for class
            c on input X[i] outputted from the last layer of your network.
        - layer_output: Dictionary containing output of each layer BEFORE
            nonlinear activation. You will need these outputs for the backprop
            algorithm. You should set layer_output[i] to be the output of layer i.

        """
        self.data=X.T
        scores = None
        layer_output = {}
        #############################################################################
        # TODO: Write the forward pass, computing the class scores for the input.   #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C). Store the output of each layer BEFORE nonlinear activation  #
        # in the layer_output dictionary                                            #
        #############################################################################
        A=self.data
        
        for i in range(1,self.num_layers+1):
                  k=self.params['W' + str(i)].T
                  Z=np.dot(k,A) +self.params['b' + str(i)].reshape(-1,1)
                  layer_output[i]=Z
                  if i==self.num_layers:
                      A=Z
                      break
                  A=self.nonlinear(Z)
               
                                      
        scores=A.T
      
        return scores, layer_output


    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """

        # Compute the forward pass
        # Store the result in the scores variable, which should be an array of shape (N, C).
        scores, layer_output = self.forward(X)

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss using the scores      #
        # output from the forward function. The loss include both the data loss and #
        # L2 regularization for weights W1,...,Wk. Store the result in the variable #
        # loss, which should be a scalar. Use the Softmax classifier loss.          #
        #############################################################################
        
        stabel=np.max(scores,axis=1).reshape(-1,1)
        Q=np.exp(scores-stabel)
        r=np.sum(Q,axis=1).reshape(-1,1)
        softmax=Q/r
        
        enc = OneHotEncoder()
        k=enc.fit_transform(y.reshape(-1,1)).toarray().astype(int)
        l=np.multiply(softmax,k)
        l1=np.sum(l,axis=1)
        l2=-np.log(l1)
        loss=np.sum(l2)/X.shape[0]
        l2_reg=0
        for i in range(1,self.num_layers+1): 
            l2_reg=l2_reg+np.sum(np.square(self.params['W' + str(i)]))          
        l2_reg=(l2_reg*reg)/2
        loss=loss+l2_reg
                      
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        
         # Backward pass: compute gradients
        grads = {}
        prob=softmax.T
        dzn=prob-(k.T)
        for layer in range(0, self.num_layers):
              # Checks if the we have not gotten to the final input layer   
              if layer!=self.num_layers-1:
                val_n_plus_1= self.num_layers - layer
                val_n= self.num_layers - layer - 1
                weight_n_plus_1 = 'W'+str(val_n_plus_1)
                bias_n_plus_1 = 'b'+str(val_n_plus_1)
        
                grads[weight_n_plus_1]=(1/(X.shape[0]))*np.dot(dzn,self.nonlinear(layer_output[val_n]).T)
                grads[bias_n_plus_1]=(1/(X.shape[0]))*np.sum(dzn,axis=1)
                grads[weight_n_plus_1] += reg*self.params[weight_n_plus_1].T
                
                dzn=np.multiply(np.dot(self.params[weight_n_plus_1],dzn),self.nonlinear_grad(layer_output[val_n]))

              # Checks if the we have not gotten to the input layer
              else:
                val_n= self.num_layers - layer
                grads['W1']=(1/(X.shape[0]))*(np.dot(dzn,X))
                grads['b1']=(1/(X.shape[0]))*np.sum(dzn,axis=1)
                grads['W1'] += reg*self.params['W1'].T
            
             
        
        ###############
        ##############################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            index = np.random.choice(np.arange(num_train),size=batch_size)
            X_batch = X[index]
            y_batch = y[index]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            for i in range(1,self.num_layers+1):
                self.params['W' + str(i)] =self.params['W' + str(i)] - (learning_rate*(grads['W' + str(i)].T))
                self.params['b' + str(i)] =self.params['b' + str(i)]- (learning_rate*(grads['b' + str(i)]))
           
            
              
              
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None
        
        scores,layer_output=self.forward(X)
        y_pred=np.argmax(scores,axis=1)

        ###########################################################################
        # TODO: Implement classification prediction. You can use the forward      #
        # function you implemented                                                #
        ###########################################################################

        return y_pred


    def sigmoid(X):
        #############################################################################
        # TODO: Write the sigmoid function                                          #
        #############################################################################
           sig=1/(1+np.exp(-X))
           return sig

    def sigmoid_grad(X):
        #############################################################################
        # TODO: Write the sigmoid gradient function                                 #
        #############################################################################
            return sigmoid(X)*(1-sigmoid(X))

    def relu(X):
        #############################################################################
        #  TODO: Write the relu function                                            #
        #############################################################################
            return (X*(X>0))
        

    def relu_grad(X):
        #############################################################################
        # TODO: Write the relu gradient function                                    #
        #############################################################################
             return (1*(X>0))

def float_bin(number, places = 3): 
  
    # split() seperates whole number and decimal  
    # part and stores it in two seperate variables 
    whole, dec = str(number).split(".") 
  
    # Convert both whole number and decimal   
    # part from string type to integer type 
    whole = int(whole) 
    dec = int (dec) 
  
    # Convert the whole number part to it's 
    # respective binary form and remove the 
    # "0b" from it. 
    res = bin(whole).lstrip("0b") + "."
  
    # Iterate the number of times, we want 
    # the number of decimal places to be 
    for x in range(places): 
  
        # Multiply the decimal value by 2  
        # and seperate the whole number part 
        # and decimal part 
        whole, dec = str((decimal_converter(dec)) * 2).split(".") 
  
        # Convert the decimal part 
        # to integer again 
        dec = int(dec) 
  
        # Keep adding the integer parts  
        # receive to the result variable 
        res += whole 
  
    return res 
  
# Function converts the value passed as 
# parameter to it's decimal representation 
def decimal_converter(num):  
    while num > 1: 
        num /= 10
    return num 
  
# Driver Code
if __name__ == '__main__':
    val = NeuralNetwork(3,[2,2],3,3)
    print(val.params)

    # n = input("Enter your floating point value : \n") 
  
    # # Take user input for the number of 
    # # decimal places user want result as 
    # p = int(input("Enter the number of decimal places of the result : \n")) 
      
    # print(float_bin(n, places = p)) 