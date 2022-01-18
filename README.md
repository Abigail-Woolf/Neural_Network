# Neural Networks and Deep Learning
## Overview
Fo this assignment, the task was to analyzes the impact of each of a philanthropic firm's donations and vet potential organizations that should receive future donations. Unfortunately, there were cases in the past where organizations took the money and disappeares, so we need to identify those high risk oranizations.

## Data
## Contents
Folder | Description
-------|------------
Practice_Files | Contains notebooks comparing differenct models to neural networks, examples of preprocessing steps, etc 
Module_17_Challenge | Contains final assignment notebook and data file

## Final Analysis
In this challenge, I had to build my own machine learning model that is able to predict the success of a venture paid by the philanthropic firm. The trained model will be used to determine the future decisions of the company — only those projects likely to be a success will receive any future funding from the firm.

neural networks are a set of algorithms that are modeled after the human brain. nn contains layrs of neurons that perform individual cmputations. These computations are connected and weighted against each other until the neurons reach a final layer and retuns a resul. 
deep learning neural network. determine which organizations should recieve donations

NN can be used for classification problems, or can behave like a regression model. they can be better than traditional ML models because they can detect complex, non-linear relationships.  and can handle messy data and a lot of noise.  however they are prone to pverfitting. 
perceptron model- four major components - input values, weight coefficients for each input, bias and a net summary function - most commonly used as a ilnear binary classifier - it is supervised

activation function - mathematical function applied to the end of each neuron (or each perceptron model) that trasnforms the output to a quantitative value, which is used as an input for other layers in the NN. 
The linear function returns the sum of our weighted inputs without transformation.
The sigmoid function is identified by a characteristic S curve. It transforms the output to a range between 0 and 1.
The tanh function is also identified by a characteristic S curve; however, it transforms the output to a range between -1 and 1.
The Rectified Linear Unit (ReLU) function returns a value from 0 to infinity, so any negative input through the activation function is 0. It is the most used activation function in neural networks due to its simplifying output, but it might not be appropriate for simpler models.
The Leaky ReLU function is a “leaky” alternative to the ReLU function, whereby negative input values will return very small negative values.
We are using the Keras module in the TensorFlow library. two keras classes: Sequential and Dense
adam optimizer usees gradient descent approach to ensure that the algorithm will not get stuck on weaker classifying variables and features.
the loss function binary_crossentropy  specifically designed to evaluate a binary classification model.
evaluation metric, which measures the quality of the machine learning model. 
A good rule of thumb for a basic neural network is to have two to three times the amount of neurons in the hidden layer as the number of inputs.
There are a few means of optimizing a neural network:

Check out your input dataset.
Add more neurons to a hidden layer.
Add additional hidden layers.
Use a different activation function for the hidden layers.
Add additional epochs to the training regimen.
This concept of a multiple-layered neural network is known as a deep learning neural network.
neural nets cannot handle categorical values so I encoded and grouped them using one-hot encoding which identifies all unique column values and splits the single categorical column into a series of columns each containing information about a single unique categorical value., binary encoding. this does not work as well if there is a large amount of unique categorical values, so we can use bucketing or binning.  to reduce the number fo unique calues.
Although basic neural networks are relatively easy to conceptualize and understand, there are limitations to using a basic neural network, such as:

A basic neural network with many neurons will require more training data than other comparable statistics and machine learning models to produce an adequate model.
Basic neural networks struggle to interpret complex nonlinear numerical data, or data with many confounding factors that have hidden effects on more than one variable.
Basic neural networks are incapable of analyzing image datasets without severe data preprocessing.

To optimize the deep neural network that I created, I tested a variety of hidden layer combinations with different activation functions. Although I tried a wide variety, I did not find any improvement in my Loss or Accuracy scores. There was one instance, where the Loss and Accuracy scores returned worse values than my original model. This happened when I used the Relu activation function on the output layer. I ended up chooseing four hidden layers with 3, 3, 6, 6 neurons respectively, although I did not see any change when altering the number of neurons or hidden layers. The Loss metric I acheived was ~0.69 and the Accuracy was ~0.53. In order to imporve the overall learning, I would stick to the binary classification model, but I would manipulate the input data more. Perhaps there were outlier data points that threw off the training model. Maybe I dropped too many variables, or maybe I did not drop enough. I would experiement with mutiple combinations of variables to see if I could improve my machine learning metrics. 
<img width="797" alt="Screen Shot 2022-01-13 at 6 24 13 PM" src="https://user-images.githubusercontent.com/65195902/149440846-76e9eee2-dae7-47ba-a820-6bece405191b.png">
<img width="791" alt="Screen Shot 2022-01-13 at 6 33 58 PM" src="https://user-images.githubusercontent.com/65195902/149441762-f264cc4b-d43b-4ca6-9c9a-5c614b714a4f.png">
<img width="852" alt="Screen Shot 2022-01-13 at 6 43 49 PM" src="https://user-images.githubusercontent.com/65195902/149442685-96c086b9-35ef-42bb-abdc-4fbe45d85bc3.png">
