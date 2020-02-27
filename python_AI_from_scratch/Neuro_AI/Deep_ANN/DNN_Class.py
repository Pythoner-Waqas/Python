
import numpy as np
import matplotlib.pyplot as plt

class DNN_base:
      def __init__(self,layer_dims,num_iterations,learning_rate,weight_initialization,
                   internal_activation, output_activation,
                   L2_regularization,lambd,dropout,keep_prob,
                   optimization="GD", alpha=0.9, eps=0.0001,
                   seed=42,verbose=True,optimum_parameters=None):
            
            self.layer_dims = layer_dims
            self.weight_initialization = weight_initialization
            self.seed = seed
            self.learning_rate = learning_rate
            self.num_iterations = num_iterations
            self.verbose = verbose
            self.internal_activation = internal_activation
            self.output_activation = output_activation
            self.L2_regularization = L2_regularization
            self.lambd = lambd
            self.dropout = dropout # Dropout if true could sometimes result into nan values
            self.keep_prob = keep_prob
            self.optimum_parameters=optimum_parameters
            self.optimization = optimization
            self.alpha = alpha
            self.eps = eps
            
      @staticmethod
      def initialize_parameters_deep(layers_dims,weight_initialization,seed):
          
            np.random.seed(seed)
            parameters = {}
            # L is number of layers in the network
            L = len(layers_dims)            
            for l in range(1, L):
                  if weight_initialization == "He":
                        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])*np.sqrt(2 /layers_dims[l - 1])
                        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
                  elif weight_initialization == "random":
                        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])*0.01
                        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
            assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
            assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))
      
            return parameters
      
      @staticmethod
      def linear_forward(pre_Activation, current_W, current_b):
            
            current_Z = np.dot(current_W,pre_Activation) + current_b #A in first layer will be X
            assert(current_Z.shape == (current_W.shape[0], pre_Activation.shape[1]))
            cache_preActivation_currentW_currentB = (pre_Activation, current_W, current_b) #This cache consists of all inputs to layer
            return current_Z, cache_preActivation_currentW_currentB

      @staticmethod
      def sigmoid(current_Z):
            current_Activation = 1/(1+np.exp(-current_Z))
            cache_currentZ = current_Z    
            return current_Activation, cache_currentZ
      
      @staticmethod
      def relu(current_Z):
            current_Activation = np.maximum(0,current_Z)
            assert(current_Activation.shape == current_Z.shape)
            cache_currentZ = current_Z 
            return current_Activation, cache_currentZ

      def linear_activation_forward(self,pre_Activation, current_W, current_b, activation):
            current_Z, cache_preActivation_currentW_currentB = self.linear_forward(pre_Activation, current_W, current_b)
              
            if activation == "sigmoid":
                  current_Activation, cache_currentZ = self.sigmoid(current_Z)
              
            elif activation == "relu":
                  current_Activation, cache_currentZ = self.relu(current_Z)
              
            assert (current_Activation.shape == (current_W.shape[0], pre_Activation.shape[1]))
            cache_preActivation_currentW_currentB_currentZ = (cache_preActivation_currentW_currentB, cache_currentZ) 
            #linear_cache stores input and activation_cache stores output of this layer
            
            return current_Activation, cache_preActivation_currentW_currentB_currentZ

      def L_model_forward(self,X, parameters,dropout,keep_prob,internal_activation,output_activation):
            cache_dropout = [] #to implement dropout
            LST_cache_preActivation_currentW_currentB_currentZ = []
            L = len(parameters) // 2    # number of layers in the neural network 
            # it is divided by two because parameters consist of both w and b for each layeR
            
            # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
            current_Activation = X
            #first element of list L is basically input features
            for l in range(1, L):
                  pre_Activation = current_Activation
                  current_W = parameters['W' + str(l)] 
                  current_b = parameters['b' + str(l)]
                  activation = internal_activation
                  current_Activation, cache_preActivation_currentW_currentB_currentZ = self.linear_activation_forward(pre_Activation, current_W, current_b, activation)
                 
                  #dropout regularization
                  if dropout:
                        tmp = np.random.rand(current_Activation.shape[0],current_Activation.shape[1])
                        tmp = tmp < keep_prob  # Step 2: convert entries of tmp to 0 or 1 (using keep_prob as the threshold)
                        cache_dropout.append(tmp)
                        current_Activation = np.multiply(current_Activation, tmp) # Step 3: shut down some neurons of current_Activation
                        current_Activation /= keep_prob # Step 4: scale the value of neurons that haven't been shut down
            
                  #store cache
                  LST_cache_preActivation_currentW_currentB_currentZ.append(cache_preActivation_currentW_currentB_currentZ)
                  
            # This following step is for last L layer 
            # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
            pre_Activation = current_Activation
            current_W = parameters['W' + str(L)] 
            current_b = parameters['b' + str(L)]
            activation = output_activation
            current_Activation, cache_preActivation_currentW_currentB_currentZ = self.linear_activation_forward(pre_Activation, current_W, current_b, activation)
            #store cache
            LST_cache_preActivation_currentW_currentB_currentZ.append(cache_preActivation_currentW_currentB_currentZ)
      
            output = current_Activation
          
            assert(output.shape == (1, X.shape[1]))
          
            return output, LST_cache_preActivation_currentW_currentB_currentZ, cache_dropout
      
      @staticmethod
      def cross_entropy_cost(output, Y, parameters, lambd, L2_regularization):
            
            #following line is added to help with nan values in case of drop out 
            #because of log loss used here
            #output = np.clip(output, 1e-5, 1. - 1e-5)  
                  
            m = Y.shape[1]
            # Compute loss from output and y.
            cross_entropy_cost = (-1 / m) * np.sum(np.multiply(Y, np.log(output)) + np.multiply(1 - Y, np.log(1 - output)))
            cross_entropy_cost = np.squeeze(cross_entropy_cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
            
            #L1 penalty or lasso regression is better than l2 penalty or ridge 
            #regression to remove certain features because it makes weight
            #of certain features to zero.

            if L2_regularization:
                  L = len(parameters)//2
                  m = Y.shape[1]
                  val = 0
                  for l in range(1,L+1):
                        val += np.sum(np.square(parameters["W"+str(l)]))
                  L2_regularization_cost = (1 / m)*(lambd / 2) * val
                  cross_entropy_cost += L2_regularization_cost
            
            assert(cross_entropy_cost.shape == ())
            return cross_entropy_cost

      @staticmethod
      def sigmoid_backward(derivative_postActivation, cache_currentZ):  
            tmp = 1/(1+np.exp(-cache_currentZ))
            derivative_postActivation_with_currentZ = tmp * (1-tmp)
            derivative_currentZ = derivative_postActivation * derivative_postActivation_with_currentZ    
            #where derivative_post_Activation is derivative of loss function upto this layer 
            assert (derivative_currentZ.shape == cache_currentZ.shape) 
            return derivative_currentZ

      @staticmethod
      def relu_backward(derivative_postActivation, cache_currentZ):
            derivative_currentZ = np.array(derivative_postActivation, copy=True) 
            #where derivative_post_Activation is derivative of loss function upto this layer
            derivative_currentZ[cache_currentZ <= 0] = 0
            assert (derivative_currentZ.shape == cache_currentZ.shape)
            return derivative_currentZ

      def linear_activation_backward(self,derivative_postActivation, cache_preActivation_currentW_currentB_currentZ,activation,samples,lambd,L2_regularization):
            cache_preActivation_currentW_currentB, cache_currentZ = cache_preActivation_currentW_currentB_currentZ
          
            #as explained in above function this dZ is derivative with respect to output of next layer
            #where output is taken after applying any activation function on linear combinations
            
            if activation == "relu":
                  derivative_currentZ = self.relu_backward(derivative_postActivation, cache_currentZ)
                  
            elif activation == "sigmoid":
                  derivative_currentZ = self.sigmoid_backward(derivative_postActivation, cache_currentZ)
                  # dA is derivative of loss upto this layer output
                  # sigmoid_backward function internally outputs dA*(activation_cache*(1-activation_cache))
          
            derivative_preActivation, derivative_currentW, derivative_currentB = self.linear_backward(derivative_currentZ, cache_preActivation_currentW_currentB,samples,lambd,L2_regularization)
          
            return derivative_preActivation, derivative_currentW, derivative_currentB



      @staticmethod
      def linear_backward(derivative_currentZ, cache_preActivation_currentW_currentB,samples,lambd,L2_regularization):
            
            pre_Activation, current_W, current_b = cache_preActivation_currentW_currentB
            
            m = pre_Activation.shape[1] #we divide it by m because this currentZ/node is calculated by
            #taking weighted sum of all previous nodes/pre_activations.
            if L2_regularization==True:
                  derivative_currentW = 1./m * np.dot(derivative_currentZ,pre_Activation.T) + ((lambd /samples)*current_W)
            else:
                  derivative_currentW = 1./m * np.dot(derivative_currentZ,pre_Activation.T) + ((lambd /samples)*current_W)
            derivative_currentB = 1./m * np.sum(derivative_currentZ, axis = 1, keepdims = True)
            derivative_preActivation = np.dot(current_W.T, derivative_currentZ)
          
            assert (derivative_preActivation.shape == pre_Activation.shape)
            assert (derivative_currentW.shape == current_W.shape)
            assert (derivative_currentB.shape == current_b.shape)
          
            return derivative_preActivation, derivative_currentW, derivative_currentB

      def L_model_backward(self,output, Y, LST_cache_preActivation_currentW_currentB_currentZ,dropout,keep_prob,
                           cache_dropout,samples,lambd,L2_regularization,internal_activation,output_activation):
            
            grads = {}
            L = len(LST_cache_preActivation_currentW_currentB_currentZ) # the number of layers excluding input layer
            #m = AL.shape[1]
            Y = Y.reshape(output.shape) # after this line, Y is the same shape as AL
          
            # Initializing the backpropagation
            derivative_output = - (np.divide(Y, output) - np.divide(1 - Y, 1 - output))
            #derivative with respect to loss for log loss function
          
            # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
            derivative_postActivation  = derivative_output
            cache_preActivation_currentW_currentB_currentZ = LST_cache_preActivation_currentW_currentB_currentZ[-1] #This current_cache contains cache for final layer        
        
            derivative_preActivation, derivative_currentW, derivative_currentB = self.linear_activation_backward(derivative_postActivation,cache_preActivation_currentW_currentB_currentZ,output_activation,samples,lambd,L2_regularization)
            
            #cache_tmp is a list containing tmp array of 0,1 / on,off for second to second_last layer excluding input and output layer
            if dropout:
                  derivative_preActivation = np.multiply(derivative_preActivation, cache_dropout[-1])  # Step 1: Apply mask cache_tmp to shut down the same neurons as during the forward propagation
                  derivative_preActivation /= keep_prob                # Step 2: Scale the value of neurons that haven't been shut down
          
            grads["dA" + str(L)] = derivative_preActivation
            grads["dW" + str(L)] = derivative_currentW
            grads["db" + str(L)] = derivative_currentB
          
            for l in reversed(range(L-1)):
                  # lth layer: (RELU -> LINEAR) gradients.
                  # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
                  derivative_postActivation = grads["dA"+str(l+2)]
                  cache_preActivation_currentW_currentB_currentZ = LST_cache_preActivation_currentW_currentB_currentZ[l]
                  derivative_preActivation, derivative_currentW, derivative_currentB = self.linear_activation_backward(derivative_postActivation, cache_preActivation_currentW_currentB_currentZ,internal_activation,samples,lambd,L2_regularization)
                  if dropout and l>0:
                        #print(derivative_preActivation.shape,cache_tmp[l-1].shape)
                        derivative_preActivation = np.multiply(derivative_preActivation, cache_dropout[l-1])  
                        # Step 1: Apply mask cache_tmp to shut down the same neurons as during the forward propagation
                        derivative_preActivation = derivative_preActivation/keep_prob
                                                   
                  grads["dA" + str(l + 1)] = derivative_preActivation
                  grads["dW" + str(l + 1)] = derivative_currentW
                  grads["db" + str(l + 1)] = derivative_currentB
            
            return grads
      
      #Implementation   
      @staticmethod
      def update_parameters(parameters, grads, learning_rate,optimization,
                            alpha,eps, nu_W, nu_b, G_W, G_b):
            
            L = len(parameters) // 2 # number of layers in the neural network
            
            # Update rule for each parameter. Use a for loop.
            for l in range(L):
                
                if optimization == "GD":
                    parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
                    parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
              
                elif optimization == "GD_Momentum":
                    #nu_W = (alpha * nu_W) + (learning_rate * grads["dW" + str(l+1)])
                    #nu_b = (alpha * nu_b) + (learning_rate * grads["db" + str(l+1)])
                    #parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - nu_W
                    #parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - nu_b
        
                    print ("Currently, there is an issue with GD_Momentum")
                    print ()
                    print ("STOP AND USE, optimization = 'GD' instead")
                    
                elif optimization == "RMSprop":
                    #G_W = (alpha*G_W)+((1-alpha)*(grads["dW" + str(l+1)]**2))
                    #G_b = (alpha*G_b)+((1-alpha)*(grads["db" + str(l+1)]**2))
                    #parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - ((learning_rate/np.sqrt(G_W+eps))*grads["dW" + str(l+1)])
                    #parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - ((learning_rate/np.sqrt(G_b+eps))*grads["db" + str(l+1)])
                    
                    print ("Currently, there is an issue with RMSprop")
                    print ()
                    print ("STOP AND USE, optimization = 'GD' instead")
                    
            return parameters, nu_W, nu_b, G_W, G_b

      def fit(self,X,Y):
            
            num_iterations=self.num_iterations
            learning_rate= self.learning_rate
            layers_dims = self.layer_dims
            weight_initialization = self.weight_initialization
            seed = self.seed
            internal_activation= self.internal_activation
            output_activation = self.output_activation 
            L2_regularization = self.L2_regularization
            dropout = self.dropout
            lambd = self.lambd
            keep_prob = self.keep_prob
            print_cost = self.verbose 
            optimum_parameters = self.optimum_parameters
            optimization = self.optimization
            alpha = self.alpha
            eps = self.eps
            
            samples = X.shape[1]
            grads = {}
            costs = [] # to keep track of the loss
            #m = X.shape[1] # number of examples
    
            if optimum_parameters is not None:
                  parameters = optimum_parameters
            else:
                  parameters = self.initialize_parameters_deep(layers_dims,weight_initialization,seed)
                  
            nu_W = 0
            nu_b = 0
            G_W = 0
            G_b = 0 
                              
            # Loop (gradient descent)
            for i in range(0, num_iterations):
                  
                  # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
                  output, LST_cache_preActivation_currentW_currentB_currentZ, cache_dropout = self.L_model_forward(X, parameters,dropout,keep_prob,internal_activation,output_activation)
                  
                  # Loss
                  cost = self.cross_entropy_cost(output, Y, parameters, lambd, L2_regularization)
                  
                  # Backward propagation.
                  grads = self.L_model_backward(output, Y, LST_cache_preActivation_currentW_currentB_currentZ,
                                           dropout,keep_prob,cache_dropout,samples,lambd,L2_regularization,
                                           internal_activation,output_activation)                
                  
                  # Update parameters.
                  parameters, nu_W, nu_b, G_W, G_b = self.update_parameters(parameters, grads, learning_rate,optimization,
                                                                            alpha,eps, nu_W, nu_b, G_W, G_b)
                  
                  # Print the cost every 100 training example
                  if print_cost and i % 100 == 0:
                        print ("Cost after iteration %i: %f" % (i, cost))
                  if print_cost and i % 100 == 0:
                        costs.append(cost)
            
            # plot the loss
            plt.plot(costs)
            plt.ylabel('cost')
            plt.xlabel('iterations (per hundreds)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
            
            return parameters
      
      def predict(self,X, y, parameters):
            
            internal_activation=self.internal_activation
            output_activation=self.output_activation
            
            m = X.shape[1]
            #n = len(parameters) // 2 # number of layers in the neural network
            p = np.zeros((1,m))
          
            # Forward propagation
            dropout = False
            keep_prob = None
            probas, _, _ = self.L_model_forward(X, parameters,dropout,keep_prob,internal_activation,output_activation)
                  
            # convert probas to 0/1 predictions
            
            p = np.where(probas>=0.5,1,0)
            
            #for i in range(0, probas.shape[1]): 
             #     if probas[0,i] > 0.5:
              #          p[0,i] = 1
               #   else:
                #        p[0,i] = 0
            
            #print results
            #print ("predictions: " + str(p))
            #print ("true labels: " + str(y))
            print("Accuracy: "  + str(np.sum((p == y)/m)))    
            return p

