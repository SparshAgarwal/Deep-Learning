import java.util.*;

// Single Perceptron
class Perceptron{
	String actFunc; // Activation function to use, valid param = "rec" and "sig"
	int numIn; // number of input nodes
	int numOut; // number of nodes this perceptron's output is connected to
	Vector<Double> inputs; // Store the inputs of the current pass of the perceptron, used in backpropagation
	double weights[]; // weights of the perceptron
	double learningRate; // learningRate of the perceptron (ETA)
	double doutdnet; // Store the derivative of the activation function, used in backpropagation

	// actFunc = activation Function of the perceptron, can be either "rec" for rectified linear or "sig" for sigmoidal
	// numIn = number of input weights
	public Perceptron(int numIn, String actFunc, double learningRate, int numOut){
		if(!actFunc.equals("rec") && !actFunc.equals("sig")){
			System.err.println("Invalid activation function parameter");
			System.exit(-1);
		}

		this.actFunc = actFunc;
		this.numIn = numIn;
		this.learningRate = learningRate;
		this.doutdnet = 0;
		this.inputs = null;

		weights = new double[numIn + 1]; // +1 for bias

		// Initialize weights
		for(int i = 0; i < weights.length; i++){
			weights[i] = Lab3_wayne_sparsh.getRandomWeight(numIn + 1, numOut, actFunc == "rec");
		}
	}

	// feedForward algorithm of the perceptron
	// inputs are the input for the perceptron
	// return the output of the perceptron
	public double feedForward(Vector<Double> inputs){
		this.inputs = inputs;
		
		double net = 0;

		for(int i = 0; i < numIn; i++){
			net += inputs.get(i) * weights[i];
		}

		net += weights[numIn]; // bias

		switch(actFunc){
			case "rec":
				return recL(net);
			case "sig":
				return sigM(net);
			default:
				System.err.println("Error, shouldn't reach this line of code");
				System.exit(-1);
				return -1;
		}
	}

	// feedForward algorithm of the perceptron
	// inputs are the input for the perceptron
	// return the output of the perceptron
	// dropout also implemented, dropOut = 0.5 means 50% drop out
	public double feedForward(Vector<Double> inputs, double dropOut){
		this.inputs = inputs;

		double net = 0;

		for(int i = 0; i < numIn; i++){
			if(Math.random() >= dropOut){
				net += inputs.get(i) * weights[i];
			}
		}

		net += weights[numIn]; // bias

		switch(actFunc){
			case "rec":
				return recL(net);
			case "sig":
				return sigM(net);
			default:
				System.err.println("Error, shouldn't reach this line of code");
				System.exit(-1);
				return -1;
		}
	}

	// backpropagation algorithm for the perceptron
	// delta equals to (dError/dout)
	// Returns newDeltai = (dError/dout * dout/dnet) * wi for backpropagation
	public double[] backPropagate(double delta){
		double newDelta = delta * doutdnet;
		double deltaList[] = new double[numIn];

		// Update all weights
		for(int i = 0; i < numIn; i++){
			deltaList[i] = newDelta * weights[i];
			double ll = learningRate * newDelta * inputs.get(i);
			weights[i] -= ll;
		}

		// Update bias
		weights[numIn] -= learningRate * newDelta;

		return deltaList;
	}
	
	// Sigmoid function
	private double sigM(double x){
		double out = 1 / (1 + Math.exp(-x));
		doutdnet = out * (1 - out);
		return out;
	}

	// Rectified Linear function
	private double recL(double x){
		if(x >= 0){
			doutdnet = 1;
			return x;
		}
		else{
			doutdnet = 0;
			return 0;
		}
	}

	// Return a copy of weights of the perceptron
	public double[] getWeights(){
		return weights.clone();
	}
	
	// Set the weights of the perceptron with param (weights)
	public void setWeights(double[] weights){
		if(weights.length != numIn + 1){
			System.err.println("Wrong number of weights, setWeights fail");
			System.exit(-1);
		}

		this.weights = weights;
	}
}

// Modified Neural Network to serve as the output layer for CNN
class OutputLayer{
	int numInputs; // Number of input nodes coming into the hidden layer
	int numHiddenUnits; // Number of nodes in the hidden layer
	int numClass; // Number of nodes in the output layer
	Vector<Perceptron> hiddenLayer; // Nodes of the hidden layer
	Vector<Perceptron> outputLayer; // Nodes of the output layer
	double outputs[];
	double totalError;

	// numInputs = number of inputs into the neural network
	// numHiddenUnits = number of Perceptrons in the hidden layer
	// numClass = number of classes to classify or number of distinct label values
	// learningRate = learning rate of all the perceptrons, aka ETA
	public OutputLayer(int numInputs, int numHiddenUnits, int numClass, double learningRate){
		this.numInputs = numInputs;
		this.numHiddenUnits = numHiddenUnits;
		this.numClass = numClass;
		hiddenLayer = new Vector<Perceptron>(numHiddenUnits);

		for(int i = 0; i < numHiddenUnits; i++){
			hiddenLayer.add(new Perceptron(numInputs, "sig", learningRate, numHiddenUnits));
		}

		outputLayer = new Vector<Perceptron>(numClass);
		outputs = new double[numClass];

		for(int i = 0; i < numClass; i++){
			outputLayer.add(new Perceptron(numHiddenUnits, "sig", learningRate, 1));
		}
	}

	// Train only a single example, used for forward pass in the output layer of CNN
	// Return the deltas for the inputs, which is the output of the previous layer
	public Vector<Double> train(Vector<Double> example, int label){
		predict(example);

		double actuals[] = new double[numClass];
		actuals[label] = 1;

		// Output Layer deltas
		Vector<double[]> outputLayerDeltas = new Vector<double[]>(numClass);

		// Backpropagation of the output layer
		for(int i = 0; i < numClass; i++){
			outputLayerDeltas.add(outputLayer.get(i).backPropagate(outputs[i] - actuals[i]));
		}

		// Hidden Layer deltas
		Vector<double[]> hiddenLayerDeltas = new Vector<double[]>(numHiddenUnits);

		for(int i = 0; i < numHiddenUnits; i++){
			double delta = 0;

			for(int j = 0; j < numClass; j++){
				delta += outputLayerDeltas.get(j)[i];
			}

			hiddenLayerDeltas.add(hiddenLayer.get(i).backPropagate(delta));
		}

		// Generate Input Layer Deltas
		Vector<Double> inputLayerDeltas = new Vector<Double>(numInputs);

		for(int i = 0; i < numInputs; i++){
			double delta = 0;

			for(int j = 0; j < numHiddenUnits; j++){
				delta += hiddenLayerDeltas.get(j)[i];
			}

			inputLayerDeltas.add(delta);
		}

		return inputLayerDeltas;
	}

	public int predict(Vector<Double> inputs){
		return predict(inputs, null);
	}

	// Predict the output of the neural network for the given inputs
	public int predict(Vector<Double> inputs, double[] actuals){
		Vector<Double> hiddenLayerOutputs = new Vector<Double>();

		// Forward pass for the hidden layer
		for(int i = 0; i < numHiddenUnits; i++){
			hiddenLayerOutputs.add(i, hiddenLayer.get(i).feedForward(inputs));
		}

		// Forward pass for the output layer
		for(int i = 0; i < numClass; i++){
			outputs[i] = outputLayer.get(i).feedForward(hiddenLayerOutputs);
		}

		return classify(outputs, actuals);
	}

	public int classify(double[] outputs){
		return classify(outputs, null);
	}

	// Assumes that output is of at least length 2, binary classification
	public int classify(double[] outputs, double[] actuals){
		int idx = 0;
		double value = outputs[0];
		
		if(actuals != null){
			totalError = 0;
		}

		for(int i = 1; i < outputs.length; i++){
			if(actuals != null){
				totalError += Math.abs(outputs[i] - actuals[i]);
			}

			if(outputs[i] > value){
				value = outputs[i];
				idx = i;
			}
		}

		return idx;
	}

	public double getTotalError(){
		return totalError;
	}

	// Assumes that there's 2 layers only, the hidden layer and the output layer
	// This function returns a list that stores all the weights of the neural network
	public List<List<double[]>> exportWeights(){
		List<List<double[]>> layers = new ArrayList<List<double[]>>();
		List<double[]> hiddenWeights = new ArrayList<double[]>();
		List<double[]> outputWeights = new ArrayList<double[]>();

		for(Perceptron p : hiddenLayer){
			hiddenWeights.add(p.getWeights());
		}

		for(Perceptron p : outputLayer){
			outputWeights.add(p.getWeights());
		}

		layers.add(hiddenWeights);
		layers.add(outputWeights);

		return layers;
	}

	// Assumes that there's 2 layers only, the hidden layer and the output layer
	// Update the weights of this neural network with layers
	public void importWeights(List<List<double[]>> layers){
		List<double[]> hiddenWeights = layers.get(0);
		List<double[]> outputWeights = layers.get(1);

		if(numHiddenUnits != hiddenWeights.size()){
			System.err.println("Wrong number of hidden Perceptron when importing weight for the hidden layer");
			System.exit(-1);
		}

		if(numClass != outputWeights.size()){
			System.err.println("Wrong number of output Perceptron when importing weight for the output layer");
			System.exit(-1);
		}

		for(int i = 0; i < numHiddenUnits; i++){
			hiddenLayer.get(i).setWeights(hiddenWeights.get(i));
		}

		for(int i = 0; i < numClass; i++){
			outputLayer.get(i).setWeights(outputWeights.get(i));
		}
	}

	// Print out the weights of each perceptron
	public void debugWeights(){
		System.out.println("HiddenLayer:");

		for(Perceptron p : hiddenLayer){
			System.out.println(Arrays.toString(p.getWeights()));
		}

		System.out.println("OutputLayer:");

		for(Perceptron p : outputLayer){
			System.out.println(Arrays.toString(p.getWeights()));
		}
	}
}

class PoolingMap{
	int windowX, windowY, outputXLength, outputYLength;
	int doutdnet[][];

	public PoolingMap(int windowXY, int outputXLength, int outputYLength){
		this.windowX = windowXY;
		this.windowY = windowXY;
		this.outputXLength = outputXLength;
		this.outputYLength = outputYLength;
	}

	public void feedForward(double[][][] inputVectors, double[][][] outputVectors, int pos){
		doutdnet = new int[outputXLength * windowX][outputYLength * windowY];

		for(int i = 0; i < outputXLength; i++){
			for(int j = 0; j < outputYLength; j++){
				int ii = i * windowX;
				int jj = j * windowY;

				double outMax = inputVectors[ii][jj][pos];
				int outX = ii;
				int outY = jj;

				for(int x = 0; x < windowX; x++){
					for(int y = 0; y < windowY; y++){
						int xx = ii + x;
						int yy = jj + y;
						if(inputVectors[xx][yy][pos] > outMax){
							outMax = inputVectors[xx][yy][pos];
							outX = xx;
							outY = yy;
						}
					}
				}
				outputVectors[i][j][pos] = outMax;
				doutdnet[outX][outY] = 1;
			}
		}
	}

	// deltas should have size outputXLength and outputYLength
	// outputDeltas should have size outputXLength * windowX and outputYLength * windowY
	// At 3rd dimension = pos
	public void backPropagate(double[][][] deltas, double[][][] outputDeltas, int pos){
		for(int i = 0; i < outputXLength; i++){
			for(int j = 0; j < outputYLength; j++){
				int ii = i * windowX;
				int jj = j * windowY;

				for(int x = 0; x < windowX; x++){
					for(int y = 0; y < windowY; y++){
						outputDeltas[ii + x][jj + y][pos] = deltas[i][j][pos] * doutdnet[ii + x][jj + y];
					}
				}
			}
		}
	}
}

class ConvolutionMap{
	int windowX, windowY, windowZ, inputXLength, inputYLength, outputXLength, outputYLength;
	double weights[][][], inputVectors[][][];
	double bias, learningRate;
	double doutdnet[][];

	// windowXY is the length of the 1st and 2nd dimension, the 3rd dimension corresponds to whether its RGB or Grayscale
	// For now windowX = windowY, so its a square, might upgrade this in the future (so that a rectangle can fit)
	public ConvolutionMap(int windowXY, int windowZ, double learningRate, int outputXLength, int outputYLength){
		this.windowX = windowXY;
		this.windowY = windowXY;
		this.windowZ = windowZ;
		this.inputXLength = outputXLength - 1 + windowX;
		this.inputYLength = outputYLength - 1 + windowY;
		this.learningRate = learningRate;
		this.outputXLength = outputXLength;
		this.outputYLength = outputYLength;

		weights = new double[windowX][windowY][windowZ];
		int totalWeights = windowX * windowY * windowZ + 1;
		
		for(int i = 0; i < windowX; i++){
			for(int j = 0; j < windowY; j++){
				for(int k = 0; k < windowZ; k++){
					weights[i][j][k] = Lab3_wayne_sparsh.getRandomWeight(totalWeights, 1, false);
				}
			}
		}

		bias = Lab3_wayne_sparsh.getRandomWeight(totalWeights, 1, false);
		doutdnet = new double[outputXLength][outputYLength];
	}

	// inputVectors = 3d input vector of dimension inputXLength and inputYLength, the 3rd dimension must be the same as the 3rd dimension of the weights
	// outputVectors = 3d output vectors of dimension outputXLength and outputYLength with 3rd dimension = pos
	// Update the vectors instead of returning the output
	public void feedForward(double[][][] inputVectors, double[][][] outputVectors, int pos){
		this.inputVectors = inputVectors;

		// Loop through output vectors
		for(int i = 0; i < outputXLength; i++){
			for(int j = 0; j < outputYLength; j++){
				// Loop through weights
				for(int x = 0; x < windowX; x++){
					for(int y = 0; y < windowY; y++){
						for(int z = 0; z < windowZ; z++){
							outputVectors[i][j][pos] += inputVectors[i + x][j + y][z] * weights[x][y][z];
						}
					}		
				}
				outputVectors[i][j][pos] += bias;
				outputVectors[i][j][pos] = sigM(outputVectors[i][j][pos]);
				doutdnet[i][j] = outputVectors[i][j][pos] * (1 - outputVectors[i][j][pos]);
			}
		}
	}

	// BackPropagate without output deltas
	public void backPropagate(double[][][] deltas, int pos){
		backPropagate(deltas, null, pos);
	}

	// BackPropagate using deltas with 3rd dimension = pos
	public void backPropagate(double[][][] deltas, double[][][] outputDeltas, int pos){
		double [][][] weightDeltas = new double[windowX][windowY][windowZ];

		// Update delta with doutdnet first
		for(int i = 0; i < outputXLength; i++){
			for(int j = 0; j < outputYLength; j++){
				deltas[i][j][pos] *= doutdnet[i][j];
			}
		}

		// Loop through weights
		for(int x = 0; x < windowX; x++){
			for(int y = 0; y < windowY; y++){
				for(int z = 0; z < windowZ; z++){
					// Loop through deltas
					for(int i = 0; i < outputXLength; i++){
						for(int j = 0; j < outputYLength; j++){
							weightDeltas[x][y][z] += inputVectors[x + i][j + y][z] * deltas[i][j][pos];
						}
					}
				}
			}
		}

		// Update weights
		for(int x = 0; x < windowX; x++){
			for(int y = 0; y < windowY; y++){
				for(int z = 0; z < windowZ; z++){
					weights[x][y][z] -= learningRate * weightDeltas[x][y][z];
				}
			}
		}

		// Update bias
		double biasDelta = 0;
		for(int i = 0; i < outputXLength; i++){
			for(int j = 0; j < outputYLength; j++){
				biasDelta += deltas[i][j][pos];
			}
		}
		bias -= learningRate * biasDelta;

		// Generate output delta
		if(outputDeltas != null){
			int alteredXLength = inputXLength + windowX - 1;
			int alteredYLength = inputYLength + windowY - 1;
			int padX = windowX - 1;
			int padY = windowY - 1;

			double alteredDeltas[][] = new double[alteredXLength][alteredYLength];

			for(int i = 0; i < outputXLength; i++){
				for(int j = 0; j < outputYLength; j++){
					alteredDeltas[i + padX][j + padY] = deltas[i][j][pos];
				}
			}

			for(int i = 0; i < inputXLength; i++){
				for(int j = 0; j < inputYLength; j++){
					// Loop through weights
					for(int x = 0; x < windowX; x++){
						for(int y = 0; y < windowY; y++){
							for(int z = 0; z < windowZ; z++){
								outputDeltas[i][j][z] += alteredDeltas[i + x][j + y] * weights[windowX - 1 - x][windowY - 1 - y][z];
							}
						}		
					}
				}
			}
		}
	}

	public double[][][] exportWeights(){
		return weights.clone();
	}

	public void importWeights(double[][][] weights){
		this.weights = weights;
	}

	// Print weights for debugging
	public void printWeights(){
		for(int i = 0; i < windowX; i++){
			for(int j = 0; j < windowY; j++){
				System.out.print("[");
				for(int k = 0; k < windowZ; k++){
					System.out.printf("%5.2f ", weights[i][j][k]);
				}
				System.out.println("]");
			}
			System.out.println("\n");
		}
		System.out.println("Shared bias: " + bias + "\n");
	}

	public double sigM(double x){
		return 1 / (1 + Math.exp(-x));
	}
}

class Utility{
	public static Vector<Double> convert3Dto1D(double[][][] vector3D){
		return convert3Dto1D(vector3D, vector3D.length, vector3D[0].length, vector3D[0][0].length);
	}

	public static Vector<Double> convert3Dto1D(double[][][] vector3D, int xLength, int yLength, int zLength){
		Vector<Double> vector1D = new Vector<Double>(xLength * yLength * zLength);

		for(int i = 0; i < xLength; i++){
			for(int j = 0; j < yLength; j++){
				for(int k = 0; k < zLength; k++){
					vector1D.add(vector3D[i][j][k]);
				}
			}	
		}
		return vector1D;
	}

	public static double[][][] convert1Dto3D(Vector<Double> vector1D, int xLength, int yLength, int zLength){
		double vector3D[][][] = new double[xLength][yLength][zLength];
		
		if(xLength * yLength * zLength != vector1D.size()){
			System.out.println("1D has vector size: " + vector1D.size());
			System.out.printf("3D Dimensions are %d %d %d\n", xLength, yLength, zLength);
			System.err.println("Wrong dimension size, can't convert 1D vector to 3D vector!!");
			System.exit(-1);
		}

		int l = 0;

		for(int i = 0; i < xLength; i++){
			for(int j = 0; j < yLength; j++){
				for(int k = 0; k < zLength; k++){
					vector3D[i][j][k] = vector1D.get(l);
					l++;
				}
			}
		}

		return vector3D;
	}
}

// Convolutional Neural Network
class CNNetwork{
	// Input variables
	int inputXLength, inputYLength, inputZLength;
	int labelSize;
	double learningRate;

	// Layer 1
	ConvolutionMap convolutionLayer1[];
	int layer1ZLength, layer1WeightSize, layer1TotalParams;
	double[][][] layer1Output; // Store the output of layer 1
	int layer1XLength, layer1YLength; // Size of layer 1 output

	// Layer 2
	PoolingMap poolingLayer1[];
	int layer2WeightSize, layer2XLength, layer2YLength, layer2ZLength, layer2TotalParams;
	double[][][] layer2Output;

	// Layer 3
	ConvolutionMap convolutionLayer2[];
	int layer3ZLength, layer3WeightSize, layer3TotalParams;
	double[][][] layer3Output;
	int layer3XLength, layer3YLength;

	// Layer 4
	PoolingMap poolingLayer2[];
	int layer4WeightSize, layer4XLength, layer4YLength, layer4ZLength, layer4TotalParams;
	double[][][] layer4Output;

	// Output Layer
	OutputLayer outputLayer;

	// Calculating error
	double totalError;

	// Optimized weights
	double[][][][] conv1Weights;
	double[][][][] conv2Weights; 
	List<List<double[]>> outputWeights;

	public CNNetwork(int xLength, int yLength, int zLength, int labelSize, double learningRate){
		this.inputXLength= xLength;
		this.inputYLength = yLength;
		this.inputZLength = zLength;
		this.labelSize = labelSize;
		this.learningRate = learningRate;

		// Layer 1
		layer1ZLength = 20; // number of feature maps of the convolutional layer
		layer1WeightSize = 5;
		layer1XLength = inputXLength - layer1WeightSize + 1;
		layer1YLength = inputYLength - layer1WeightSize + 1;
		layer1TotalParams = layer1XLength * layer1YLength * layer1ZLength;
		layer1Output = new double[layer1XLength][layer1YLength][layer1ZLength];

		convolutionLayer1 = new ConvolutionMap[layer1ZLength];
		conv1Weights = new double[layer1ZLength][layer1WeightSize][layer1WeightSize][inputZLength];

		for(int i = 0; i < layer1ZLength; i++){
			convolutionLayer1[i] = new ConvolutionMap(layer1WeightSize, inputZLength, learningRate, layer1XLength, layer1YLength);
		}

		printLayer(1, "Convolutional Layer", inputXLength, inputYLength, inputZLength, layer1XLength, layer1YLength, layer1ZLength);

		// Layer 2
		layer2WeightSize = 2;
		layer2XLength = layer1XLength / layer2WeightSize;
		layer2YLength = layer1YLength / layer2WeightSize;
		layer2ZLength = layer1ZLength;
		layer2TotalParams = layer2XLength * layer2YLength * layer2ZLength;
		layer2Output = new double[layer2XLength][layer2YLength][layer2ZLength];

		poolingLayer1 = new PoolingMap[layer2ZLength];

		for(int i = 0; i < layer2ZLength; i++){
			poolingLayer1[i] = new PoolingMap(layer2WeightSize, layer2XLength, layer2YLength);
		}

		printLayer(2, "Pooling Layer", layer1XLength, layer1YLength, layer1ZLength, layer2XLength, layer2YLength, layer2ZLength);

		// Layer 3
		layer3ZLength = 20; // number of feature maps of the convolutional layer
		layer3WeightSize = 5;
		layer3XLength = layer2XLength - layer3WeightSize + 1;
		layer3YLength = layer2YLength - layer3WeightSize + 1;
		layer3TotalParams = layer3XLength * layer3YLength * layer3ZLength;
		layer3Output = new double[layer3XLength][layer3YLength][layer3ZLength];

		convolutionLayer2 = new ConvolutionMap[layer3ZLength];
		conv2Weights = new double[layer3ZLength][layer3WeightSize][layer3WeightSize][layer2ZLength];

		for(int i = 0; i < layer3ZLength; i++){
			convolutionLayer2[i] = new ConvolutionMap(layer3WeightSize, layer2ZLength, learningRate, layer3XLength, layer3YLength);
		}

		printLayer(3, "Convolutional Layer", layer2XLength, layer2YLength, layer2ZLength, layer3XLength, layer3YLength, layer3ZLength);

		// Layer 4
		layer4WeightSize = 2;
		layer4XLength = layer3XLength / layer4WeightSize;
		layer4YLength = layer3YLength / layer4WeightSize;
		layer4ZLength = layer3ZLength;
		layer4TotalParams = layer4XLength * layer4YLength * layer4ZLength;
		layer4Output = new double[layer4XLength][layer4YLength][layer4ZLength];

		poolingLayer2 = new PoolingMap[layer4ZLength];

		for(int i = 0; i < layer4ZLength; i++){
			poolingLayer2[i] = new PoolingMap(layer4WeightSize, layer4XLength, layer4YLength);
		}

		printLayer(4, "Pooling Layer", layer3XLength, layer3YLength, layer3ZLength, layer4XLength, layer4YLength, layer4ZLength);

		// Output Layer
		int numHiddenUnits = 150;
		outputLayer = new OutputLayer(layer4TotalParams, numHiddenUnits, labelSize, learningRate);

		printLayer(5, "Output Layer", layer4XLength, layer4YLength, layer4ZLength, labelSize, 1, 1);
	}

	public double train(Vector<CNNExample> featureVectors){
		for(CNNExample e : featureVectors){
			// Forward Pass
			// Layer 1: Convolutional Layer
			for(int i = 0; i < layer1ZLength; i++){
				convolutionLayer1[i].feedForward(e.example, layer1Output, i);
			}

			// Layer 2: Pooling Layer (Max)
			for(int i = 0; i < layer2ZLength; i++){
				poolingLayer1[i].feedForward(layer1Output, layer2Output, i);
			}

			// Layer 3: Convolutional Layer
			for(int i = 0; i < layer3ZLength; i++){
				convolutionLayer2[i].feedForward(layer2Output, layer3Output, i);
			}

			// Layer 4: Pooling Layer (Max)
			for(int i = 0; i < layer4ZLength; i++){
				poolingLayer2[i].feedForward(layer3Output, layer4Output, i);
			}

			// Output Layer
			Vector<Double> outputLayerDeltas = outputLayer.train(Utility.convert3Dto1D(layer4Output), e.label);
			double outputLayer3DDeltas[][][] = Utility.convert1Dto3D(outputLayerDeltas, layer4XLength, layer4YLength, layer4ZLength);

			// Backward Pass
			double poolingLayer2Deltas[][][] = new double[layer3XLength][layer3YLength][layer3ZLength];

			// Layer 4: Pooling Layer (Max)
			for(int i = 0; i < layer4ZLength; i++){
				poolingLayer2[i].backPropagate(outputLayer3DDeltas, poolingLayer2Deltas, i);
			}

			double convolutionLayer2Deltas[][][] = new double[layer2XLength][layer2YLength][layer2ZLength];

			// Layer 3: Convolutional Layer
			for(int i = 0; i < layer3ZLength; i++){
				convolutionLayer2[i].backPropagate(poolingLayer2Deltas, convolutionLayer2Deltas, i);
			}

			double poolingLayer1Deltas[][][] = new double[layer1XLength][layer1YLength][layer1ZLength];

			// Layer 2: Pooling Layer (Max)
			for(int i = 0; i < layer2ZLength; i++){
				poolingLayer1[i].backPropagate(convolutionLayer2Deltas, poolingLayer1Deltas, i);
			}

			double convolutionLayer1Deltas[][][] = new double[inputXLength][inputYLength][inputZLength];

			// Layer 1: Convolutional Layer
			for(int i = 0; i < layer1ZLength; i++){
				convolutionLayer1[i].backPropagate(poolingLayer1Deltas, i);
			}
		}

		return 0;
	}

	public int predict(double[][][] example){
		return predict(example, null);
	}

	public int predict(double[][][] example, double[] actuals){
		// Forward Pass
		// Layer 1: Convolutional Layer
		for(int i = 0; i < layer1ZLength; i++){
			convolutionLayer1[i].feedForward(example, layer1Output, i);
		}

		// Layer 2: Pooling Layer (Max)
		for(int i = 0; i < layer2ZLength; i++){
			poolingLayer1[i].feedForward(layer1Output, layer2Output, i);
		}

		// Layer 3: Convolutional Layer
		for(int i = 0; i < layer3ZLength; i++){
			convolutionLayer2[i].feedForward(layer2Output, layer3Output, i);
		}

		// Layer 2: Pooling Layer (Max)
		for(int i = 0; i < layer4ZLength; i++){
			poolingLayer2[i].feedForward(layer3Output, layer4Output, i);
		}

		return outputLayer.predict(Utility.convert3Dto1D(layer4Output), actuals);
	}

	// Return the accuracy of the test
	public double test(Vector<CNNExample> examples){
		return test(examples, false);
	}

	// Return the accuracy of the test, print out results if debug is True
	public double test(Vector<CNNExample> examples, Boolean debug){
		int numCorrect = 0;
		totalError = 0;

		// For calculating confusion matrix
		int givenLabel[] = new int[examples.size()];
		int outputLabel[] = new int[examples.size()];
		int k = 0;

		for(CNNExample e : examples){
			int label = e.label;
			
			double actuals[] = new double[labelSize];
			actuals[label] = 1;
			
			if(debug){
				givenLabel[k] = label;
				outputLabel[k] = predict(e.example, actuals);
				totalError += outputLayer.getTotalError();
				k ++;
			}
			else{
				if(predict(e.example) == label){
					numCorrect ++;
				}
			}
		}

		if(debug){
			numCorrect = confusionMatrix(givenLabel, outputLabel);
		}

		double acc = numCorrect / (double)examples.size();
		double error = (totalError / labelSize) / examples.size();

		if(debug){
			System.out.printf("Accuracy: %.4f%%\nMean Squared Error: %.4f%%\n", acc * 100, error * 100);
		}
		
		return acc;
	}

	public void storeOptimalWeights(){
		for(int i = 0; i < layer1ZLength; i++){
			conv1Weights[i] = convolutionLayer1[i].exportWeights();
		}

		for(int i = 0; i < layer3ZLength; i++){
			conv2Weights[i] = convolutionLayer2[i].exportWeights();
		}

		outputWeights = outputLayer.exportWeights();
	}

	public void setOptimalWeights(){
		for(int i = 0; i < layer1ZLength; i++){
			convolutionLayer1[i].importWeights(conv1Weights[i]);
		}

		for(int i = 0; i < layer3ZLength; i++){
			convolutionLayer2[i].importWeights(conv2Weights[i]);
		}

		outputLayer.importWeights(outputWeights);
	}

	// Print out the confusion matrix and return the number of correctly predicted labels
	// Length of x and y should be the same
	public int confusionMatrix(int actual[], int predicted[]){
		int correct = 0;
		int matrix[][] = new int[labelSize][labelSize];
		String sep = "";

		for(int i = 0; i < actual.length; i++){
			matrix[actual[i]][predicted[i]] ++;
		}

		for(int i = 0; i < labelSize; i++){
			correct += matrix[i][i];
			sep += "------";
		}
		
		System.out.println("---------------- Confusion Matrix ----------------");
		System.out.println(sep);
		for(int i = 0; i < labelSize; i++){
			System.out.print("|");
			for(int j = 0; j < labelSize; j++){
				System.out.printf("%4d |", matrix[j][i]);
			}
			System.out.println("\n" + sep);
		}

		return correct;
	}

	private void printLayer(int layerIdx, String name, int inX, int inY, int inZ, int outX, int outY, int outZ){
		System.out.printf("Layer%d: %s\nInput Dimensions: %d x %d x %d\nOutput Dimensions: %d x %d x %d\n",
			layerIdx, name, inX, inY, inZ, outX, outY, outZ);
	}
}

class CNNExample{
	public double[][][] example;
	public int label;

	public CNNExample(double [][][] example, int label){
		this.example = example;
		this.label = label;
	}
}

public class CNNClassifier{
	CNNetwork cnn;
	Boolean isRGB;
	int labelSize;
	double learningRate;
	int xLength, yLength, zLength; // x = height of image (rows), y = width of image (cols), z = 4 if (RGB) else 1 (Grayscale)

	// Assumes that we are training images with the same length and width
	// inputVectorSize = length of the width and height of the input vector
	// isRGB = whether the vectors are in RGB, length of 3rd dimension = 4 if RGB, 1 if not RGB (grayscale)
	public CNNClassifier(int length, int width, Boolean isRGB, int labelSize, double learningRate){
		this.xLength = length;
		this.yLength = width;
		zLength = isRGB ? 4 : 1;
		this.labelSize = labelSize;
		this.learningRate = learningRate;
		this.isRGB = isRGB;

		cnn = new CNNetwork(xLength, yLength, zLength, labelSize, learningRate);
	}

	public void train(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors, Vector<Vector<Double>> testFeatureVectors, int patience, int epochStep, Boolean debug){
		Vector<CNNExample> trainExamples = bulkConvert1Dto3D(trainFeatureVectors);
		Vector<CNNExample> tuneExamples = bulkConvert1Dto3D(tuneFeatureVectors);
		Vector<CNNExample> testExamples = bulkConvert1Dto3D(testFeatureVectors);

		long  overallStart = System.currentTimeMillis(), start = overallStart;
		double bestAcc = cnn.test(tuneExamples, debug);
		int epoch = 0;
		int bestTuneEpoch = 0;

		for(int i = 0; i < patience; i++){
			if(debug){
				System.out.println("Epoch: " + epoch);
			}

			// Train in batch before tuning, if epochStep == 1, then its train once and follow by a tune
			for(int j = 0; j < epochStep; j++){
				Lab3_wayne_sparsh.permute(trainExamples);
				cnn.train(trainExamples);
			}

			// Get train set accuracy
			System.out.println("~~~~Trainset~~~~");
			// Output the results
			double acctrain = cnn.test(trainExamples, debug);
			System.out.printf("%.4f%%%n",100*acctrain);

			// Get tune set accuracy
			System.out.println("~~~~Tuneset~~~~");
			// Output the results
			double acctune = cnn.test(tuneExamples, debug);
			System.out.printf("%.4f%%%n",100*acctune);

			// Get test set accuracy
			System.out.println("~~~~Testset~~~~");
			// Output the results
			double acctest = cnn.test(testExamples, debug);
			System.out.printf("%.4f%%%n",100*acctest);

			if(acctune > bestAcc){
				bestAcc = acctune;
				i = -1;
				bestTuneEpoch = epoch;

				// Keep track of the optimal weights
				cnn.storeOptimalWeights();
			}

			System.out.println("Done with Epoch # " + Lab3_wayne_sparsh.comma(epoch) + ".  Took " + Lab3_wayne_sparsh.convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + Lab3_wayne_sparsh.convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
  			start = System.currentTimeMillis();

			epoch ++;
		}

		cnn.setOptimalWeights();

		System.out.printf("\nBest Tuning Set Accuracy: %.4f%% at Epoch: %d\n", bestAcc * 100, bestTuneEpoch);
	}

	public double test(Vector<Vector<Double>> featureVectors){
		return test(featureVectors, false);
	}

	public double test(Vector<Vector<Double>> featureVectors, Boolean debug){
		Vector<CNNExample> examples = bulkConvert1Dto3D(featureVectors);
		return cnn.test(examples, debug);
	}

	public double predict(Vector<Double> example){
		CNNExample cnnexample = convert1Dto3D(example);
		return cnn.predict(cnnexample.example);
	}

	public Vector<CNNExample> bulkConvert1Dto3D(Vector<Vector<Double>> vectors1D){
		Vector<CNNExample> examples = new Vector<CNNExample>(vectors1D.size());

		for(int i = 0; i < vectors1D.size(); i++){
			examples.add(convert1Dto3D(vectors1D.get(i)));
		}

		return examples;
	}

	// Use xLength, yLength and zLength to convert the 1Dvector
	public CNNExample convert1Dto3D(Vector<Double> vector1D){
		double vector3D[][][] = new double[xLength][yLength][zLength];
		int l = 0;

		for(int i = 0; i < xLength; i++){
			for(int j = 0; j < yLength; j++){
				for(int k = 0; k < zLength; k++){
					vector3D[i][j][k] = vector1D.get(l);
					l++;
				}
			}
		}

		return new CNNExample(vector3D, vector1D.get(l).intValue());
	}
}