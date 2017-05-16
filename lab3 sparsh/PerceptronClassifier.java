import java.util.Random;
import java.util.Vector;

// Single Perceptron
class SPerceptron{
	String actFunc; // Activation function to use, valid param = "rec" and "sig"
	int numIn; // number of input nodes
	Vector<Double> inputs; // Store the inputs of the current pass of the perceptron, used in backpropagation
	double weights[]; // weights of the perceptron
	double learningRate; // learningRate of the perceptron (ETA)
	double doutdnet; // Store the derivative of the activation function, used in backpropagation

	// actFunc = activation Function of the perceptron, can be either "rec" for rectified linear or "sig" for sigmoidal
	// numIn = number of input weights
	public SPerceptron(int numIn, String actFunc, double learningRate){
		if(!actFunc.equals("rec") && !actFunc.equals("sig")){
			System.err.println("Invalid activation function parameter");
			System.exit(-1);
		}

		this.actFunc = actFunc;
		this.numIn = numIn;
		this.learningRate = learningRate;
		this.doutdnet = 0;
		this.inputs = null;

		weights = new double[numIn + 1];

		// Initialize weights
		for(int i = 0; i < weights.length; i++){
			weights[i] = Lab3_wayne_sparsh.getRandomWeight(numIn + 1, 1, actFunc == "rec");
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

	// backpropagation algorithm for the perceptron
	// delta equals to (dError/dout)
	// Returns newDeltai = (dError/dout * dout/dnet) * wi for backpropagation
	public void backPropagate(double delta){
		double newDelta = delta * doutdnet;

		// Update all weights
		for(int i = 0; i < numIn; i++){
			weights[i] -= learningRate * delta * doutdnet * inputs.get(i);
		}

		// Update bias
		weights[numIn] -= learningRate * delta * doutdnet;
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

public class PerceptronClassifier{
	int inputVectorSize;
	int labelSize;
	double learningRate;
	double totalError;
	Vector<SPerceptron> perceptrons;

	public PerceptronClassifier(int inputVectorSize, int labelSize, double learningRate){
		this.inputVectorSize = inputVectorSize; // inputVectorSize includes the bias
		this.labelSize = labelSize;
		this.learningRate = learningRate;
		perceptrons = new Vector<SPerceptron>(labelSize);

		for(int i = 0; i < labelSize; i++){
			perceptrons.add(new SPerceptron(inputVectorSize - 1, "sig", learningRate));
		}
	}

	// Train the classifier
	public void train(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors, int patience, int epochStep, Boolean debug){
		// long  overallStart = System.currentTimeMillis(), start = overallStart;
		int epoch = 0;
		double bestAcc = test(tuneFeatureVectors);
		int bestTuneEpoch = 0;
		Vector<double[]> bestWeights = new Vector<double[]>(labelSize);

		for(int i = 0; i < labelSize; i++){
			bestWeights.add(null);
		}

		for(int i = 0; i < patience; i++){
			// Train in batch before tuning, if epochStep == 1, then its train once and follow by a tune
			if(debug){
				System.out.println("Epoch: " + epoch);
			}

			for(int j = 0; j < epochStep; j++){
				Lab3_wayne_sparsh.permute(trainFeatureVectors);

				for(Vector<Double> example : trainFeatureVectors){
					double actuals[] = new double[labelSize];
					actuals[(example.lastElement().intValue())] = 1;

					for(int k = 0; k < labelSize; k++){
						perceptrons.get(k).backPropagate(perceptrons.get(k).feedForward(example) - actuals[k]);
					}
				}
			}

			// Test if accuracy increase, if yes, then keep track of the best accuracy
			double acc = test(tuneFeatureVectors, debug);
			
			if(acc > bestAcc){
				i = -1;
				bestAcc = acc;
				bestTuneEpoch = epoch;

				for(int p = 0; p < labelSize; p++){
					bestWeights.set(p, perceptrons.get(p).getWeights());
				}
			}

			// System.out.println("Done with Epoch # " + Lab3_wayne_sparsh.comma(epoch) + ".  Took " + Lab3_wayne_sparsh.convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + Lab3_wayne_sparsh.convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
        	// start = System.currentTimeMillis();

			for(int p = 0; p < labelSize; p++){
				perceptrons.get(p).setWeights(bestWeights.get(p));
			}

        	epoch ++;
		}

		System.out.printf("\nBest Tuning Set Accuracy: %.4f%% at Epoch: %d\n", bestAcc * 100, bestTuneEpoch);
	}

	// Return the accuracy of the classifier based on the given featureVectors
	public double test(Vector<Vector<Double>> featureVectors){
		return test(featureVectors, false);
	}

	// Return the accuracy of the classifier based on the given featureVectors
	public double test(Vector<Vector<Double>> featureVectors, Boolean debug){
		totalError = 0;
		int numCorrect = 0;

		for(Vector<Double> example: featureVectors){
			double actuals[] = new double[labelSize];
			actuals[(example.lastElement().intValue())] = 1;

			// if(debug && true){
				// System.out.println(example.lastElement().intValue() + ":" + (int)predict(example));
			// }

			if(actuals[predict(example, actuals)] == 1){
				numCorrect += 1;
			}
		}

		double accuracy = numCorrect / (double)featureVectors.size();
		double error = (totalError / labelSize) / featureVectors.size();

		if(debug){
			System.out.printf("Accuracy: %.4f%%\nAverage Error: %.4f%%\n", accuracy * 100, error * 100);
		}

		return accuracy;
	}

	// Predict the output only, use the other overloaded method for calculating total error
	public double predict(Vector<Double> example){
		return predict(example, null);
	}

	// Predict output and keep track of the error at the same time using actuals[] = actual output
	public int predict(Vector<Double> example, double[] actuals){
		double bestOutput = -1;
		int bestLabel = -1;

		for(int i = 0; i < labelSize; i++){
			double output = perceptrons.get(i).feedForward(example);

			if(actuals != null){
				totalError += Math.abs(output - actuals[i]);	
			}

			if(output > bestOutput){
				bestOutput = output;
				bestLabel = i;
			}
		}

		return bestLabel;
	}
}