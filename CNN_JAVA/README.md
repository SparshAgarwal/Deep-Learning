# CNN_JAVA

This folder contains a convolutional neural network written from scratch in JAVA.  
This is written for a Deep Neural Network class. The code is written to be easily understandable than efficient.  

Classes in CNNClassifier.java:  
1) CNNetwork: Convolutional Neural Network, controls all the layers, input, output, feedforward, backpropagation  
2) NeuralNetwork: A complete neural network, used for the hiddenlayer and outputlayer of the CNN  
3) Perceptron: A single Perceptron used in the NeuralNetwork class  
4) PoolingMap: A single pooling filter, multiple filters combine to form the pooling layer, currently uses MAX pooling  
5) ConvolutionMap: A single convolutional filter inside the convolutional layer  

These classes are stored for future purposes in case one has to write some neural network from scratch. These codes can be used as a good base. The code are easy for someone to visualize the feedforward and backpropagation mechanism of a neural network.  