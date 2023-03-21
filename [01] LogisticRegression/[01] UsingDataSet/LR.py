# Logistic Regression using excel file

##1) Importing Modules

##Import necessary packages

#One Series
import numpy as np
import matplotlib.pyplot as plt
import h5py

#Second One
import scipy
from PIL import Image
from scipy import ndimage

%matplotlib inline

##2) Loading & Exploring Datasets

#Befor U run this module U need to add Cat.py into U're GoogleDrop Directory, if using the Colab IDE

#Our do it like see in bellow:

from google.colab import drive
drive.mount('/content/drive/')

import os
for dirname, _, filenames in os.walk('/content/drive/MyDrive/LR-TAofML/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from Cats import load_dataset

#Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#Detail the data (cat/non-cat)

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

#Example of a cat

index = 27
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]))
print ("it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

#And Example of a not-cat

index = 101
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]))
print ("it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

#Reshape the training and test examples

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

##3) Standardize Data Set

#standardization of dataset.

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

print(100*np.sum(train_set_y == 0)/len(train_set_y[0]),'% of Non-cat in the training data')
print(100*np.sum(train_set_y == 1)/len(train_set_y[0]),'% of Cat in the training data')

##4) Implement Necessary Functions

###If U need any more detail in this Sec, see HW-02.pdf

###4-1) Sigmoid Function:

#Compute the sigmoid of z

def sigmoid(z):
  
    s = 1 / (1 + np.exp(-z))
    
    return s

###4-2) Initialization of Parameters

#creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

def initialize_with_zeros(dim):
  
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

###4-3) Calculation of Derivatives and Cost Function:

#Forward and Backward propagation
#cost function Implementation and its gradient for the propagation explained above

def propagate(w, b, X, Y):
    
    m = X.shape[1]
    
    #FORWARD PROPAGATION (FROM X TO COST)

    A = sigmoid(np.dot(w.T, X) + b)       
    #compute activation

    cost = -1/m * np.sum(Y*np.log(A) + (1 - Y)*np.log(1 - A))
    #compute cost
    
    #BACKWARD PROPAGATION (TO FIND GRAD)
    dw = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

###4-4) Gradient Descent Algorithm: 

#Optimization: Updating the parameters using Gradient Descent
#optimizes w and b by running a gradient descent algorithm

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
  
    costs = []
    
    for i in range(num_iterations):
        
        
        #Cost and gradient calculation 
        grads, cost = propagate(w, b, X, Y)
        
        #Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        #update rule
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        #Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        #Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

###4-5) Predict Function:

#Predict: Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5)
#Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

def predict(w, b, X):
  
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    #Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        
        #Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[:, i] <= 0.5:
            Y_prediction[:, i] = 0
        else:
            Y_prediction[:, i] = 1
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

###4-6) Merge of All Functions into Model Function:

#Training the Model: merge all functions into a model
#Builds logistic regression model by calling the function That we've implemented in previouse Sections

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    
    #initialize parameters with zeros
    dim = X_train.shape[0]
    w, b = initialize_with_zeros(dim)

    #Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations= num_iterations, learning_rate = learning_rate, print_cost = print_cost)
    
    #Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    #Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    #Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train": Y_prediction_train, 
         "w": w, 
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    
    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

d

d['w'] #optimal value of d. It is a vector of size (12288, 1)

d['b'] #optimal value of d. It is a real number.

##5) Plot learning curve (with costs)

#Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

##6) Choice of Learning Rate

learning_rates = [0.01]
#learning_rates = 0.01, 0.001, 0.0001


models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))

    #num_iterations = 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

#Report on the effects of learning rate and the maximum number of iterations for train and test are discribe in Report.pdf

##Contact Us: Benyaminteymuri@gmail.com

#The End of Code