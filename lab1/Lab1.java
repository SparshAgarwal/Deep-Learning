import java.io.*;
import java.util.*;

// Stores info of the input files
// Number of Features, Feature Names, Feature Values and Label Values
class DataInfo{
	int numFeatures;
	String[] featureNames;
	List<List<String>> featureValues;
	String[] labelValues;

	public DataInfo(CommentScanner scn){
		// Get number of features
		if(scn.hasNext()){
			numFeatures = scn.nextInt();
			if(numFeatures == -1){
				System.err.println("Invalid file format for file: " + scn.getFileName());
				System.exit(-1);	
			}
		}
		else{
			System.err.println("Invalid file format for file: " + scn.getFileName());
			System.exit(-1);
		}

		// Get feature names and feature values
		featureNames = new String[numFeatures];
		featureValues = new ArrayList<List<String>>();

		for(int i = 0; i < numFeatures; i++){
			if(scn.hasNext()){
				String[] nextString = scn.next().split("-");

				if(nextString.length != 2){
					System.err.println("Invalid file format for file: " + scn.getFileName());	
					System.exit(-1);
				}

				featureNames[i] = nextString[0].trim();

				String[] values = nextString[1].trim().split(" ");
				featureValues.add(new ArrayList<String>());

				for(int j = 0; j < values.length; j++){
					featureValues.get(i).add(values[j].trim());
				}
			}
			else{
				System.err.println("Invalid file format for file: " + scn.getFileName());
				System.exit(-1);
			}			
		}

		// Get label values, assume boolean labels
		labelValues = new String[2];

		for(int i = 0; i < 2; i++){
			if(scn.hasNext()){
				labelValues[i] = scn.next();
			}
			else{
				System.err.println("Invalid file format for file: " + scn.getFileName());
				System.exit(-1);
			}
		}
	}

	// Print out DataInfo
	public void print(){
		System.out.println("Number of features: " + numFeatures);

		System.out.println("Features:");
		for(int i = 0; i < numFeatures; i++){
			System.out.print("\t" + featureNames[i] + "-> ");

			for(int j = 0; j < featureValues.get(i).size() - 1; j++){
				System.out.print(featureValues.get(i).get(j) + ", ");
			}

			System.out.println(featureValues.get(i).get(featureValues.get(i).size() - 1));
		}

		System.out.println("Labels: " + labelValues[0] + ", " + labelValues[1]);
	}

	// Return the index of the value for the feature[featureIndex], return -1 if not found
	public int getFeatureValueIndex(int featureIndex, String value){
		if(featureIndex >= numFeatures){
			return -1;
		}

		return featureValues.get(featureIndex).indexOf(value);
	}

	// Return the label index, return -1 if not found
	public int getLabelIndex(String label){
		if(label.equals(labelValues[0])){
			return 0;
		}
		else if(label.equals(labelValues[1])){
			return 1;
		}
		else{
			return -1;
		}
	}

	public String indexToLabel(int x){
		if(x < 0 || x >= 2){
			System.err.println("Label index out of bounds.");
			System.exit(-1);
		}

		return labelValues[x];
	}
}

// Scanner that ignore comments
class CommentScanner{
	private Scanner scn;
	private	String nextWord;
	private boolean nextExist;
	private String fileName;

	public CommentScanner(String fileName){
		this.fileName = fileName;
		File inFile = new File(fileName);
		try{
			scn = new Scanner(inFile).useDelimiter("//.*\n|\n");
		} catch (FileNotFoundException ex){
			System.err.println("Error: File " + fileName + " is not found.");
			System.exit(-1);
		}

		nextExist = true;
		setNext();
	}

	private void setNext(){
		while(scn.hasNext()){
			nextWord = scn.next().trim();

			if(!nextWord.equals("")){
				return;
			}
		}
		nextExist = false;
	}

	// Return true if there is a valid string in the buffer
	public boolean hasNext(){
		return nextExist;
	}

	// Return the next integer
	public int nextInt(){
		if(!nextExist){
			System.err.println("Scanner has no words left.");
			System.exit(-1);
		}

		int x = -1;

		try{
			x = Integer.parseInt(nextWord);
		} catch (NumberFormatException ex){
		}
		
		setNext();

		return x;
	}

	// Return the next string
	public String next(){
		if(!nextExist){
			System.err.println("Scanner has no words left.");
			System.exit(-1);
		}

		String x = nextWord;
		setNext();

		return x;
	}

	// Return the file name of the file this scanner is reading from
	public String getFileName(){
		return fileName;
	}
}

// Data structure to load and store examples
class ExampleList{
	// First item is the label while the rest are features
	int[][] examples;
	int numExamples;

	public ExampleList(CommentScanner scn, DataInfo di){
		// Set up array
		if(scn.hasNext()){
			numExamples = scn.nextInt();
			if(numExamples == -1){
				System.err.println("Invalid file format for file: " + scn.getFileName());
				System.exit(-1);	
			}
		}
		else{
			System.err.println("Invalid file format for file: " + scn.getFileName());
			System.exit(-1);
		}

		examples = new int[numExamples][di.numFeatures + 1];

		// Load examples
		for(int i = 0; i < numExamples; i++){
			if(scn.hasNext()){
				String[] nextString = scn.next().split("\\s+");

				if(nextString.length != di.numFeatures + 2){
					System.err.println("Invalid file format for file: " + scn.getFileName());	
					System.exit(-1);
				}
				
				// Get examples
				for(int j = 0; j < di.numFeatures; j++){
					examples[i][j] = di.getFeatureValueIndex(j, nextString[j + 2]);
					
					if(examples[i][j] == -1){
						System.err.println("Invalid feature value: " + nextString[j + 2]);
						System.exit(-1);
					}					
				}

				// Get Label
				examples[i][di.numFeatures] = di.getLabelIndex(nextString[1]); // Label

				if(examples[i][di.numFeatures] == -1){
					System.err.println("Invalid label: " + nextString[1]);
					System.exit(-1);
				}
			}
			else{
				System.err.println("Invalid file format for file: " + scn.getFileName());
				System.exit(-1);
			}
		}
	}

	// Print out all examples
	public void print(){
		for(int i = 0; i < numExamples; i++){
			System.out.print(i + 1 + ": ");

			for(int j = 0; j < examples[0].length - 1; j++){
				System.out.print(examples[i][j] + ", ");
			}

			System.out.println(examples[i][examples[0].length - 1]);
		}
	}
}

// Single Perceptron
class Perceptron{
	DataInfo di;
	double weights[];
	double learningRate;

	public Perceptron(DataInfo di){
		new Perceptron(di, 0.1);
	}

	public Perceptron(DataInfo di, double learningRate){
		this.di = di;
		this.learningRate = learningRate;

		Random rdm = new Random();
		weights = new double[di.numFeatures + 1];

		for(int i = 0; i < di.numFeatures + 1; i++){
			weights[i] = rdm.nextDouble();
		}
	}

	// Get the weights of this perceptron
	public double[] getWeights(){
		return weights;
	}

	// Set the weights of this perceptron
	public void setWeights(double[] x){
		weights = x;
	}

	// Train the perceptron with examples
	public void train(ExampleList ex){
		for(int i = 0; i < ex.numExamples; i++){
			double diff = learningRate * (ex.examples[i][weights.length - 1] - predict(ex.examples[i]));

			for(int j = 0; j < weights.length - 1; j++){
				weights[j] += diff * ex.examples[i][j];
			}

			weights[weights.length - 1] += diff;
		}
	}

	// Predict the value of an example
	public int predict(int[] row){
		double sum = 0;

		for(int i = 0; i < row.length; i++){
			sum += row[i] * weights[i];
		}

		sum += weights[weights.length - 1];

		if(sum >= 0){
			return 1;
		}
		else{
			return 0;
		}
	}

	// Prediction on an entire set of examples
	public double test(ExampleList ex){
		double sum = 0;

		for(int i = 0; i < ex.numExamples; i++){
			if(predict(ex.examples[i]) == ex.examples[i][ex.examples[0].length - 1]){
				sum += 1;
			}
		}

		return sum / ex.numExamples;
	}

	// Prediction on an entire set of examples with output
	public double testWithOutput(ExampleList ex){
		double sum = 0;

		for(int i = 0; i < ex.numExamples; i++){
			int prediction = predict(ex.examples[i]);
			if(prediction == ex.examples[i][ex.examples[0].length - 1]){
				sum += 1;
			}
			System.out.println(di.indexToLabel(prediction));
		}
		
		return sum / ex.numExamples;	
	}

	// Set the learning rate of the perceptron
	public void setLearningRate(double x){
		learningRate = x;
	}
}	

public class Lab1{
	// Check for correct program arguments
	public static void checkArgs(String[] args){
		if(args.length != 3){
			System.err.println("Usage: Lab1 <fileNameOfTrain> <fileNameOfTune> <fileNameOfTest>");
			System.exit(-1);
		}
	}

	public static void main(String[] args){
		checkArgs(args);

		// Scanner that ignore comments for files
		CommentScanner trainScn = new CommentScanner(args[0]);
		CommentScanner tuneScn = new CommentScanner(args[1]);
		CommentScanner testScn = new CommentScanner(args[2]);

		// Load metadata of the files
		DataInfo dataInfo = new DataInfo(trainScn);
		new DataInfo(tuneScn);
		new DataInfo(testScn);

		// Load examples
		ExampleList trainEx = new ExampleList(trainScn, dataInfo);
		ExampleList tuneEx = new ExampleList(tuneScn, dataInfo);
		ExampleList testEx = new ExampleList(testScn, dataInfo);

		// Initialize perceptrons
		Perceptron perceptron = new Perceptron(dataInfo, 0.1);

		// Train y=numPerceptrons and pick the best one
		// Use early stopping for each perceptron
		// Early stopping rule: Stop if accuracy does not increase after x=patience epoch
		int numPerceptrons = 50;
		int patience = 30;

		for(int i = 0; i < numPerceptrons; i++){
			Perceptron perceptronNew = new Perceptron(dataInfo, 0.1);

			double acc = perceptronNew.test(tuneEx);
			double weights[] = perceptronNew.getWeights();

			for(int j = 0; j < patience; j++){
				perceptronNew.train(trainEx);
				double newAcc = perceptronNew.test(tuneEx);

				if(newAcc > acc){
					acc = newAcc;
					j = 0;
					weights = perceptronNew.getWeights();
				}

				perceptronNew.setWeights(weights);
			}

			if(perceptronNew.test(tuneEx) > perceptron.test(tuneEx)){
				perceptron = perceptronNew;
			}
		}

		// Print out result and overall accuracy
		System.out.printf("Overall Accuracy: %.2f\n", perceptron.testWithOutput(testEx) * 100);
	}
}