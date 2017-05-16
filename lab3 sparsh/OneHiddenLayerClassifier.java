import java.util.*;

// Neural Network that consists of a hidden layer and an output layer
class NeuralNetwork{
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
	public NeuralNetwork(int numInputs, int numHiddenUnits, int numClass, double learningRate){
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

	// Train the neural network using examples
	public void train(Vector<Vector<Double>> examples){
		for(Vector<Double> example : examples){

			double label = example.lastElement();
			predict(example); // Feedforward

			double actuals[] = new double[numClass];
			actuals[(int)label] = 1;

			List<double[]> deltas = new ArrayList<double[]>(); // To store the derivative of the error

			// Backpropagation of the output layer
			for(int i = 0; i < numClass; i++){
				deltas.add(outputLayer.get(i).backPropagate(outputs[i] - actuals[i]));
			}

			// Backpropagation of the hidden layer 
			for(int i = 0; i < numHiddenUnits; i++){
				double delta = 0;

				for(int j = 0; j < numClass; j++){
					delta += deltas.get(j)[i];
				}

				hiddenLayer.get(i).backPropagate(delta);
			}
		}
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

	// Return the accuracy of the test
	public double test(Vector<Vector<Double>> examples){
		return test(examples, false);
	}

	// Return the accuracy of the test, print out results if debug is True
	public double test(Vector<Vector<Double>> examples, Boolean debug){
		int numCorrect = 0;
		totalError = 0;

		// For calculating confusion matrix
		int givenLabel[] = new int[examples.size()];
		int outputLabel[] = new int[examples.size()];
		int k = 0;

		for(Vector<Double> example : examples){
			int label = example.lastElement().intValue();
			

			double actuals[] = new double[numClass];
			actuals[label] = 1;
			
			int output = predict(example, actuals);

			if(debug){
				givenLabel[k] = label;
				outputLabel[k] = output;
				k ++;
			}
			else{
				if(output == label){
					numCorrect ++;
				}
			}
		}

		if(debug){
			numCorrect = confusionMatrix(givenLabel, outputLabel);
		}

		double acc = numCorrect / (double)examples.size();
		double error = (totalError / numClass) / examples.size();

		if(debug){
			System.out.printf("Accuracy: %.4f%%\nMean Squared Error: %.4f%%\n", acc * 100, error * 100);
		}
		
		return acc;
	}

	// Print out the confusion matrix and return the number of correctly predicted labels
	// Length of x and y should be the same
	public int confusionMatrix(int actual[], int predicted[]){
		int correct = 0;
		int matrix[][] = new int[numClass][numClass];
		String sep = "";

		for(int i = 0; i < actual.length; i++){
			matrix[actual[i]][predicted[i]] ++;
		}

		for(int i = 0; i < numClass; i++){
			correct += matrix[i][i];
			sep += "------";
		}
		
		System.out.println("---------------- Confusion Matrix ----------------");
		System.out.println(sep);
		for(int i = 0; i < numClass; i++){
			System.out.print("|");
			for(int j = 0; j < numClass; j++){
				System.out.printf("%4d |", matrix[j][i]);
			}
			System.out.println("\n" + sep);
		}

		return correct;
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

// Main class
public class OneHiddenLayerClassifier{
	NeuralNetwork nn;
	int inputVectorSize;
	int numHiddenUnits;
	int labelSize;
	double learningRate;

	public OneHiddenLayerClassifier(int inputVectorSize, int numHiddenUnits, int labelSize, double learningRate){
		this.inputVectorSize = inputVectorSize;
		this.numHiddenUnits = numHiddenUnits;
		this.labelSize = labelSize;
		this.learningRate = learningRate;

		nn = new NeuralNetwork(inputVectorSize - 1, numHiddenUnits, labelSize, learningRate);
	}

	public void train(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors, int patience, int epochStep, Boolean debug){
		long  overallStart = System.currentTimeMillis(), start = overallStart;
		double bestAcc = test(tuneFeatureVectors, debug);
		List<List<double[]>> optimalWeights = nn.exportWeights();
		int epoch = 0;
		int bestTuneEpoch = 0;

		for(int i = 0; i < patience; i++){
			if(debug){
				System.out.println("Epoch: " + epoch);
			}

			// Train in batch before tuning, if epochStep == 1, then its train once and follow by a tune
			for(int j = 0; j < epochStep; j++){
				Lab3_wayne_sparsh.permute(trainFeatureVectors);
				nn.train(trainFeatureVectors);
			}

			// Get tune set accuracy
			System.out.println("~~~~Tuneset~~~~");
			double acc = test(tuneFeatureVectors, debug);
			
			if(acc > bestAcc){
				bestAcc = acc;
				i = -1;
				bestTuneEpoch = epoch;

				// Keep track of the optimal weights
				optimalWeights = nn.exportWeights();
			}

			System.out.println("Done with Epoch # " + Lab3_wayne_sparsh.comma(epoch) + ".  Took " + Lab3_wayne_sparsh.convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + Lab3_wayne_sparsh.convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
        	start = System.currentTimeMillis();

			epoch ++;
		}

		nn.importWeights(optimalWeights);

		System.out.printf("\nBest Tuning Set Accuracy: %.4f%% at Epoch: %d\n", bestAcc * 100, bestTuneEpoch);
	}

	public double test(Vector<Vector<Double>> featureVectors){
		return test(featureVectors, false);
	}

	public double test(Vector<Vector<Double>> featureVectors, Boolean debug){
		return nn.test(featureVectors, debug);
	}

	public double predict(Vector<Double> example){
		return nn.predict(example);
	}
}