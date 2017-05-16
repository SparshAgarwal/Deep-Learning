/**
 * @Author: Yuting Liu and Jude Shavlik.  
 * 
 * Copyright 2017.  Free for educational and basic-research use.
 * 
 * The main class for Lab of cs638/838.
 * 
 * Reads in the image files and stores BufferedImage's for every example.  Converts to fixed-length
 * feature vectors (of doubles).  Can use RGB (plus grey-scale) or use grey scale.
 * 
 * You might want to debug and experiment with your Deep ANN code using a separate class, but when you turn in Lab.java, insert that class here to simplify grading.
 * 
 * Some snippets from Jude's code left in here - feel free to use or discard.
 *
 */

import java.util.*;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.Vector;

import javax.imageio.ImageIO;

public class Lab {
    
    private static int     imageSize = 32; // Images are imageSize x imageSize.  The provided data is 128x128, but this can be resized by setting this value (or passing in an argument).  
                                           // You might want to resize to 8x8, 16x16, 32x32, or 64x64; this can reduce your network size and speed up debugging runs.
                                           // ALL IMAGES IN A TRAINING RUN SHOULD BE THE *SAME* SIZE.
    private static enum    Category { airplanes, butterfly, flower, grand_piano, starfish, watch };  // We'll hardwire these in, but more robust code would not do so.
    
    private static final Boolean    useRGB = true; // If true, FOUR units are used per pixel: red, green, blue, and grey.  If false, only ONE (the grey-scale value).
    private static       int unitsPerPixel = (useRGB ? 4 : 1); // If using RGB, use red+blue+green+grey.  Otherwise just use the grey value.
            
    private static String    modelToUse = "deep"; // Should be one of { "perceptrons", "oneLayer", "deep" };  You might want to use this if you are trying approaches other than a Deep ANN.
    private static int       inputVectorSize;         // The provided code uses a 1D vector of input features.  You might want to create a 2D version for your Deep ANN code.  
                                                      // Or use the get2DfeatureValue() 'accessor function' that maps 2D coordinates into the 1D vector.  
                                                      // The last element in this vector holds the 'teacher-provided' label of the example.

    private static double eta       =    0.01, fractionOfTrainingToUse = 1.00, dropoutRate = 0.00; // To turn off drop out, set dropoutRate to 0.0 (or a neg number).
    private static int    maxEpochs = 1000; // Feel free to set to a different value.

    public static void main(String[] args) {
        String trainDirectory = "images/trainset/";
        String  tuneDirectory = "images/tuneset/";
        String  testDirectory = "images/testset/";
        
        if(args.length > 5) {
            System.err.println("Usage error: java Lab <train_set_folder_path> <tune_set_folder_path> <test_set_folder_path> <imageSize>");
            System.exit(1);
        }
        if (args.length >= 1) { trainDirectory = args[0]; }
        if (args.length >= 2) {  tuneDirectory = args[1]; }
        if (args.length >= 3) {  testDirectory = args[2]; }
        if (args.length >= 4) {  imageSize     = Integer.parseInt(args[3]); }
    
        // Here are statements with the absolute path to open images folder
        File trainsetDir = new File(trainDirectory);
        File tunesetDir  = new File( tuneDirectory);
        File testsetDir  = new File( testDirectory);
        
        // create three datasets
        Dataset trainset = new Dataset();
        Dataset  tuneset = new Dataset();
        Dataset  testset = new Dataset();
        
        // Load in images into datasets.
        long start = System.currentTimeMillis();
        loadDataset(trainset, trainsetDir);
        System.out.println("The trainset contains " + comma(trainset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        start = System.currentTimeMillis();
        loadDataset(tuneset, tunesetDir);
        System.out.println("The  testset contains " + comma( tuneset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        start = System.currentTimeMillis();
        loadDataset(testset, testsetDir);
        System.out.println("The  tuneset contains " + comma( testset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        
        // Now train a Deep ANN.  You might wish to first use your Lab 2 code here and see how one layer of HUs does.  Maybe even try your perceptron code.
        // We are providing code that converts images to feature vectors.  Feel free to discard or modify.
        start = System.currentTimeMillis();
        trainANN(trainset, tuneset, testset);
        System.out.println("\nTook " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " to train.");
        
    }

    public static void loadDataset(Dataset dataset, File dir) {
        for(File file : dir.listFiles()) {
            // check all files
             if(!file.isFile() || !file.getName().endsWith(".jpg")) {
                continue;
            }
            //String path = file.getAbsolutePath();
            BufferedImage img = null, scaledBI = null;
            try {
                // load in all images
                img = ImageIO.read(file);
                // every image's name is in such format:
                // label_image_XXXX(4 digits) though this code could handle more than 4 digits.
                String name = file.getName();
                int locationOfUnderscoreImage = name.indexOf("_image");
                
                // Resize the image if requested.  Any resizing allowed, but should really be one of 8x8, 16x16, 32x32, or 64x64 (original data is 128x128).
                if (imageSize != 128) {
                    scaledBI = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB);
                    Graphics2D g = scaledBI.createGraphics();
                    g.drawImage(img, 0, 0, imageSize, imageSize, null);
                    g.dispose();
                }
                
                Instance instance = new Instance(scaledBI == null ? img : scaledBI, name, name.substring(0, locationOfUnderscoreImage));

                dataset.add(instance);
            } catch (IOException e) {
                System.err.println("Error: cannot load in the image file");
                System.exit(1);
            }
        }
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////
    
    private static Category convertCategoryStringToEnum(String name) {
        if ("airplanes".equals(name))   return Category.airplanes; // Should have been the singular 'airplane' but we'll live with this minor error.
        if ("butterfly".equals(name))   return Category.butterfly;
        if ("flower".equals(name))      return Category.flower;
        if ("grand_piano".equals(name)) return Category.grand_piano;
        if ("starfish".equals(name))    return Category.starfish;
        if ("watch".equals(name))       return Category.watch;
        throw new Error("Unknown category: " + name);       
    }
    
    public static double getRandomWeight(int fanin, int fanout, boolean isaReLU) { // This is one, 'rule of thumb' for initializing weights.
        double range = Math.max(10 * Double.MIN_VALUE,
                                isaReLU ? 2.0        / Math.sqrt(fanin + fanout)    // From paper by Glorot & Bengio.  See http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
                                        : 4.0 * Math.sqrt(6.0 / (fanin + fanout))); // See http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
        return (2.0 * random() - 1.0) * range;
    }

    
    // Map from 2D coordinates (in pixels) to the 1D fixed-length feature vector.
    private static double get2DfeatureValue(Vector<Double> ex, int x, int y, int offset) { // If only using GREY, then offset = 0;  Else offset = 0 for RED, 1 for GREEN, 2 for BLUE, and 3 for GREY.
        return ex.get(unitsPerPixel * (y * imageSize + x) + offset); // Jude: I have not used this, so might need debugging.
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////////

    
    // Return the accuracy of the testset for the choosen model
    private static double trainANN(Dataset trainset, Dataset tuneset, Dataset testset) {
        Instance sampleImage = trainset.getImages().get(0); // Assume there is at least one train image!
        inputVectorSize = sampleImage.getWidth() * sampleImage.getHeight() * unitsPerPixel + 1; // The '-1' for the bias is not explicitly added to all examples (instead code should implicitly handle it).  The final 1 is for the CATEGORY.
        
        // For RGB, we use FOUR input units per pixel: red, green, blue, plus grey.  Otherwise we only use GREY scale.
        // Pixel values are integers in [0,255], which we convert to a double in [0.0, 1.0].
        // The last item in a feature vector is the CATEGORY, encoded as a double in 0 to the size on the Category enum.
        // We do not explicitly store the '-1' that is used for the bias.  Instead code (to be written) will need to implicitly handle that extra feature.
        System.out.println("\nThe input vector size is " + comma(inputVectorSize - 1) + ".\n");
        
        Vector<Vector<Double>> trainFeatureVectors = new Vector<Vector<Double>>(trainset.getSize());
        Vector<Vector<Double>>  tuneFeatureVectors = new Vector<Vector<Double>>( tuneset.getSize());
        Vector<Vector<Double>>  testFeatureVectors = new Vector<Vector<Double>>( testset.getSize());
        
        long start = System.currentTimeMillis();
        fillFeatureVectors(trainFeatureVectors, trainset);
        System.out.println("Converted " + trainFeatureVectors.size() + " TRAIN examples to feature vectors. Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        start = System.currentTimeMillis();
        fillFeatureVectors( tuneFeatureVectors,  tuneset);
        System.out.println("Converted " +  tuneFeatureVectors.size() + " TUNE  examples to feature vectors. Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        start = System.currentTimeMillis();
        fillFeatureVectors( testFeatureVectors,  testset);
        System.out.println("Converted " +  testFeatureVectors.size() + " TEST  examples to feature vectors. Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        System.out.println("\nTime to start learning!");
        
        // Call your Deep ANN here.  We recommend you create a separate class file for that during testing and debugging, but before submitting your code cut-and-paste that code here.
        
        if      ("perceptrons".equals(modelToUse)) return trainPerceptrons(trainFeatureVectors, tuneFeatureVectors, testFeatureVectors); // This is optional.  Either comment out this line or just right a 'dummy' function.
        else if ("oneLayer".equals(   modelToUse)) return trainOneHU(      trainFeatureVectors, tuneFeatureVectors, testFeatureVectors); // This is optional.  Ditto.
        else if ("deep".equals(       modelToUse)) return trainDeep(       trainFeatureVectors, tuneFeatureVectors, testFeatureVectors);
        return -1;
    }
    
    private static void fillFeatureVectors(Vector<Vector<Double>> featureVectors, Dataset dataset) {
        for (Instance image : dataset.getImages()) {
            featureVectors.addElement(convertToFeatureVector(image));
        }
    }

    private static Vector<Double> convertToFeatureVector(Instance image) {
        Vector<Double> result = new Vector<Double>(inputVectorSize);        

        for (int index = 0; index < inputVectorSize - 1; index++) { // Need to subtract 1 since the last item is the CATEGORY.
            if (useRGB) {
                int xValue = (index / unitsPerPixel) % image.getWidth(); 
                int yValue = (index / unitsPerPixel) / image.getWidth();
            //  System.out.println("  xValue = " + xValue + " and yValue = " + yValue + " for index = " + index);
                if      (index % unitsPerPixel == 0) result.add(image.getRedChannel()  [xValue][yValue] / 255.0); // If unitsPerPixel > 4, this if-then-elseif needs to be edited!
                else if (index % unitsPerPixel == 1) result.add(image.getGreenChannel()[xValue][yValue] / 255.0);
                else if (index % unitsPerPixel == 2) result.add(image.getBlueChannel() [xValue][yValue] / 255.0);
                else                     result.add(image.getGrayImage()   [xValue][yValue] / 255.0); // Seems reasonable to also provide the GREY value.
            } else {
                int xValue = index % image.getWidth();
                int yValue = index / image.getWidth();
                result.add(                         image.getGrayImage()   [xValue][yValue] / 255.0);
            }
        }
        result.add((double) convertCategoryStringToEnum(image.getLabel()).ordinal()); // The last item is the CATEGORY, representing as an integer starting at 0 (and that int is then coerced to double).
        
        return result;
    }
    
    ////////////////////  Some utility methods (cut-and-pasted from JWS' Utils.java file). ///////////////////////////////////////////////////
    
    private static final long millisecInMinute = 60000;
    private static final long millisecInHour   = 60 * millisecInMinute;
    private static final long millisecInDay    = 24 * millisecInHour;
    public static String convertMillisecondsToTimeSpan(long millisec) {
        return convertMillisecondsToTimeSpan(millisec, 0);
    }
    public static String convertMillisecondsToTimeSpan(long millisec, int digits) {
        if (millisec ==    0) { return "0 seconds"; } // Handle these cases this way rather than saying "0 milliseconds."
        if (millisec <  1000) { return comma(millisec) + " milliseconds"; } // Or just comment out these two lines?
        if (millisec > millisecInDay)    { return comma(millisec / millisecInDay)    + " days and "    + convertMillisecondsToTimeSpan(millisec % millisecInDay,    digits); }
        if (millisec > millisecInHour)   { return comma(millisec / millisecInHour)   + " hours and "   + convertMillisecondsToTimeSpan(millisec % millisecInHour,   digits); }
        if (millisec > millisecInMinute) { return comma(millisec / millisecInMinute) + " minutes and " + convertMillisecondsToTimeSpan(millisec % millisecInMinute, digits); }
        
        return truncate(millisec / 1000.0, digits) + " seconds"; 
    }

    public static String comma(int value) { // Always use separators (e.g., "100,000").
        return String.format("%,d", value);     
    }    
    public static String comma(long value) { // Always use separators (e.g., "100,000").
        return String.format("%,d", value);     
    }   
    public static String comma(double value) { // Always use separators (e.g., "100,000").
        return String.format("%,f", value);     
    }
    public static String padLeft(String value, int width) {
        String spec = "%" + width + "s";
        return String.format(spec, value);      
    }
    
    /**
     * Format the given floating point number by truncating it to the specified
     * number of decimal places.
     * 
     * @param d
     *            A number.
     * @param decimals
     *            How many decimal places the number should have when displayed.
     * @return A string containing the given number formatted to the specified
     *         number of decimal places.
     */
    public static String truncate(double d, int decimals) {
        double abs = Math.abs(d);
        if (abs > 1e13)             { 
            return String.format("%."  + (decimals + 4) + "g", d);
        } else if (abs > 0 && abs < Math.pow(10, -decimals))  { 
            return String.format("%."  +  decimals      + "g", d);
        }
        return     String.format("%,." +  decimals      + "f", d);
    }
    
    /** Randomly permute vector in place.
     *
     * @param <T>  Type of vector to permute.
     * @param vector Vector to permute in place. 
     */
    public static <T> void permute(Vector<T> vector) {
        if (vector != null) { // NOTE from JWS (2/2/12): not sure this is an unbiased permute; I prefer (1) assigning random number to each element, (2) sorting, (3) removing random numbers.
            // But also see "http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle" which justifies this.
            /*  To shuffle an array a of n elements (indices 0..n-1):
                                    for i from n - 1 downto 1 do
                                    j <- random integer with 0 <= j <= i
                                    exchange a[j] and a[i]
             */

            for (int i = vector.size() - 1; i >= 1; i--) {  // Note from JWS (2/2/12): to match the above I reversed the FOR loop that Trevor wrote, though I don't think it matters.
                int j = random0toNminus1(i + 1);
                if (j != i) {
                    T swap =    vector.get(i);
                    vector.set(i, vector.get(j));
                    vector.set(j, swap);
                }
            }
        }
    }
    
    public static Random randomInstance = new Random();  // Change the 638 * 838 to get a different sequence of random numbers.
    
    /**
     * @return The next random double.
     */
    public static double random() {
        return randomInstance.nextDouble();
    }

    /**
     * @param lower
     *            The lower end of the interval.
     * @param upper
     *            The upper end of the interval. It is not possible for the
     *            returned random number to equal this number.
     * @return Returns a random integer in the given interval [lower, upper).
     */
    public static int randomInInterval(int lower, int upper) {
        return lower + (int) Math.floor(random() * (upper - lower));
    }


    /**
     * @param upper
     *            The upper bound on the interval.
     * @return A random number in the interval [0, upper).
     * @see Utils#randomInInterval(int, int)
     */
    public static int random0toNminus1(int upper) {
        return randomInInterval(0, upper);
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////////  Write your own code below here.  Feel free to use or discard what is provided.
        
    private static double trainPerceptrons(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors, Vector<Vector<Double>> testFeatureVectors) {
        // Set up training feature vectors
        if (fractionOfTrainingToUse < 1.0) {  // Randomize list, then get the first N of them.
            int numberToKeep = (int) (fractionOfTrainingToUse * trainFeatureVectors.size());
            Vector<Vector<Double>> trainFeatureVectors_temp = new Vector<Vector<Double>>(numberToKeep);

            permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.
            for (int i = 0; i <numberToKeep; i++) {
                trainFeatureVectors_temp.add(trainFeatureVectors.get(i));
            }
            trainFeatureVectors = trainFeatureVectors_temp;
        }
        
        // reportPerceptronConfig();

        // Set up classifier (best eta = 0.01, patience = 300) grayscale 32x32
        PerceptronClassifier classifier = new PerceptronClassifier(inputVectorSize, Category.values().length, eta);
        int patience = 300;

        System.out.printf("Using: ETA = %f, patience = %d, epochStep = %d\n", eta, patience, 1);
        classifier.train(trainFeatureVectors, tuneFeatureVectors, patience, 1, true);

        System.out.println("**************** FINAL RESULTS ****************");
        System.out.println("Test Set result for perceptrons:");
        
        return classifier.test(testFeatureVectors, true);
    }
    
    private static void reportPerceptronConfig() {
        System.out.println(  "***** PERCEPTRON: UseRGB = " + useRGB + ", imageSize = " + imageSize + "x" + imageSize + ", fraction of training examples used = " + truncate(fractionOfTrainingToUse, 2) + ", eta = " + truncate(eta, 2) + ", dropout rate = " + truncate(dropoutRate, 2));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////   ONE HIDDEN LAYER

    private static boolean debugOneLayer               = false;  // If set true, more things checked and/or printed (which does slow down the code).
    private static int    numberOfHiddenUnits          = 250;
    
    private static double trainOneHU(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors, Vector<Vector<Double>> testFeatureVectors) {
        // Set up training feature vectors
        if (fractionOfTrainingToUse < 1.0) {  // Randomize list, then get the first N of them.
            int numberToKeep = (int) (fractionOfTrainingToUse * trainFeatureVectors.size());
            Vector<Vector<Double>> trainFeatureVectors_temp = new Vector<Vector<Double>>(numberToKeep);

            permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.
            for (int i = 0; i <numberToKeep; i++) {
                trainFeatureVectors_temp.add(trainFeatureVectors.get(i));
            }
            trainFeatureVectors = trainFeatureVectors_temp;
        }

        reportOneLayerConfig();

        int numHiddenUnits = 200;
        OneHiddenLayerClassifier classifier = new OneHiddenLayerClassifier(inputVectorSize, numHiddenUnits, Category.values().length, eta);

        int patience = 30;
        int epochStep = 5;
        System.out.printf("Using: ETA = %f, numHiddenUnits = %d, patience = %d, epochStep = %d\n", eta, numHiddenUnits, patience, epochStep);
        classifier.train(trainFeatureVectors, tuneFeatureVectors, patience, epochStep, true);

        System.out.println("**************** FINAL RESULTS ****************");
        System.out.println("Test Set result for one hidden layer neural network:");
        
        return classifier.test(testFeatureVectors, true);
    }
    
    private static void reportOneLayerConfig() {
        System.out.println(  "***** ONE-LAYER: UseRGB = " + useRGB + ", imageSize = " + imageSize + "x" + imageSize + ", fraction of training examples used = " + truncate(fractionOfTrainingToUse, 2) 
                + ", eta = " + truncate(eta, 2)   + ", dropout rate = "      + truncate(dropoutRate, 2) + ", number HUs = " + numberOfHiddenUnits
            //  + ", activationFunctionForHUs = " + activationFunctionForHUs + ", activationFunctionForOutputs = " + activationFunctionForOutputs
            //  + ", # forward props = " + comma(forwardPropCounter)
                );
    //  for (Category cat : Category.values()) {  // Report the output unit biases.
    //      int catIndex = cat.ordinal();
    //
    //      System.out.print("  bias(" + cat + ") = " + truncate(weightsToOutputUnits[numberOfHiddenUnits][catIndex], 6));
    //  }   System.out.println();
    }

    // private static long forwardPropCounter = 0;  // Count the number of forward propagations performed.
    
    
    ////////////////////////////////////////////////////////////////////////////////////////////////  DEEP ANN Code


    private static double trainDeep(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors,  Vector<Vector<Double>> testFeatureVectors) {
        // Set up training feature vectors
        if (fractionOfTrainingToUse < 1.0) {  // Randomize list, then get the first N of them.
            int numberToKeep = (int) (fractionOfTrainingToUse * trainFeatureVectors.size());
            Vector<Vector<Double>> trainFeatureVectors_temp = new Vector<Vector<Double>>(numberToKeep);

            permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.
            for (int i = 0; i <numberToKeep; i++) {
                trainFeatureVectors_temp.add(trainFeatureVectors.get(i));
            }
            trainFeatureVectors = trainFeatureVectors_temp;
        }

        int patience = 40; // Number of epoch = patience * epochStep to run until accuracy increases, if accuracy doesn't increase, terminate
        int epochStep = 5; // Batch Size, test the tuning set for accuracy after every epoch = epochStep

        // Output details
        System.out.println("Convolutional Neural Network Deep Training");
        System.out.println("Data Info: useRGB = " + useRGB + ", imageSize = " + imageSize + ", fractionOfTrainingToUse = " + truncate(fractionOfTrainingToUse, 2));
        System.out.printf("Using: ETA = %f, Dropout = %.2f, Patience = %d, BatchSize = %d.\n", eta, dropoutRate, patience, epochStep);
        System.out.println();

        CNNClassifier classifier = new CNNClassifier(imageSize, imageSize, useRGB, Category.values().length, eta, dropoutRate);

        classifier.train(trainFeatureVectors, tuneFeatureVectors, testFeatureVectors, patience, epochStep, true);

        System.out.println("**************** FINAL RESULTS ****************");
        System.out.println("Test Set result for convolutional neural network:");
        
        return classifier.test(testFeatureVectors, true);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////

}

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
                System.out.printf("%4d |", matrix[i][j]);
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
            weights[i] = Lab.getRandomWeight(numIn + 1, 1, actFunc == "rec");
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

class PerceptronClassifier{
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
                Lab.permute(trainFeatureVectors);

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

            // System.out.println("Done with Epoch # " + Lab.comma(epoch) + ".  Took " + Lab.convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + Lab.convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
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
            System.out.printf("Accuracy: %.4f%%\nMean Squared Error: %.4f%%\n", accuracy * 100, error * 100);
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

// Main class
class OneHiddenLayerClassifier{
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
                Lab.permute(trainFeatureVectors);
                nn.train(trainFeatureVectors);
            }

            // Get tune set accuracy
            if(debug){
                System.out.println("~~~~Tuneset~~~~");
            }
            
            double acc = test(tuneFeatureVectors, debug);
            
            if(acc > bestAcc){
                bestAcc = acc;
                i = -1;
                bestTuneEpoch = epoch;

                // Keep track of the optimal weights
                optimalWeights = nn.exportWeights();
            }

            System.out.println("Done with Epoch # " + Lab.comma(epoch) + ".  Took " + Lab.convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + Lab.convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
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
            weights[i] = Lab.getRandomWeight(numIn + 1, numOut, actFunc == "rec");
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
        if(Math.random() < dropOut){
            doutdnet = 0;
            return 0;
        }

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
    public double[] backPropagate(double delta){
        if(doutdnet == 0){
            return new double[numIn];
        }

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
    double dropout;

    // numInputs = number of inputs into the neural network
    // numHiddenUnits = number of Perceptrons in the hidden layer
    // numClass = number of classes to classify or number of distinct label values
    // learningRate = learning rate of all the perceptrons, aka ETA
    public OutputLayer(int numInputs, int numHiddenUnits, int numClass, double learningRate, double dropout){
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
            hiddenLayerOutputs.add(i, hiddenLayer.get(i).feedForward(inputs, dropout));
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

// Filters in the Pooling Layer
class PoolingMap{
    int windowX, windowY, outputXLength, outputYLength;
    int doutdnet[][];

    // WindowXY = pooling layer window, it is a square for now
    // OutputXLength and OutputYLength = dimensions of the pooling map output
    public PoolingMap(int windowXY, int outputXLength, int outputYLength){
        this.windowX = windowXY;
        this.windowY = windowXY;
        this.outputXLength = outputXLength;
        this.outputYLength = outputYLength;
    }

    // Feedforward by downsampling the input using the max function with a window of windowX x windowY
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

// Filters in the Convolutional Layer
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
                    weights[i][j][k] = Lab.getRandomWeight(totalWeights, 1, false);
                }
            }
        }

        bias = Lab.getRandomWeight(totalWeights, 1, false);
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

    // Export weights so that the optimal weights can be stored
    public double[][][] exportWeights(){
        return weights.clone();
    }

    // Load weights
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

    // Sigmoid function
    public double sigM(double x){
        return 1 / (1 + Math.exp(-x));
    }
}

// Some utility functions to flatten the matrix or convert the matrix to 3D
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
    double dropout;

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

    public CNNetwork(int xLength, int yLength, int zLength, int labelSize, double learningRate, double dropout){
        this.inputXLength= xLength;
        this.inputYLength = yLength;
        this.inputZLength = zLength;
        this.labelSize = labelSize;
        this.learningRate = learningRate;
        this.dropout = dropout;

        // Layer 1
        layer1ZLength = 10; // number of feature maps of the convolutional layer
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
        int numHiddenUnits = 50;
        outputLayer = new OutputLayer(layer4TotalParams, numHiddenUnits, labelSize, learningRate, dropout);

        printLayer(5, "Output Layer", layer4XLength, layer4YLength, layer4ZLength, labelSize, 1, 1);

        storeOptimalWeights();
    }

    // Train the Convolutional Neural Network by feedForwarding the input through layers and then back propagate the error
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

    // Classify the result
    public int predict(double[][][] example){
        return predict(example, null);
    }

    // Classify the result, actuals is used for calculating errors, set actuals = null for no debug
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

    // Import weights from the CNN and store it
    public void storeOptimalWeights(){
        for(int i = 0; i < layer1ZLength; i++){
            conv1Weights[i] = convolutionLayer1[i].exportWeights();
        }

        for(int i = 0; i < layer3ZLength; i++){
            conv2Weights[i] = convolutionLayer2[i].exportWeights();
        }

        outputWeights = outputLayer.exportWeights();
    }

    // Export weights that are stored into the CNN
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

    // Utility function to print details of the layers
    private void printLayer(int layerIdx, String name, int inX, int inY, int inZ, int outX, int outY, int outZ){
        System.out.printf("\nLayer%d: %s\nInput Dimensions: %d x %d x %d\nOutput Dimensions: %d x %d x %d\n",
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

class CNNClassifier{
    CNNetwork cnn;
    Boolean isRGB;
    int labelSize;
    double learningRate, dropout;
    int xLength, yLength, zLength; // x = height of image (rows), y = width of image (cols), z = 4 if (RGB) else 1 (Grayscale)

    // Assumes that we are training images with the same length and width
    // inputVectorSize = length of the width and height of the input vector
    // isRGB = whether the vectors are in RGB, length of 3rd dimension = 4 if RGB, 1 if not RGB (grayscale)
    public CNNClassifier(int length, int width, Boolean isRGB, int labelSize, double learningRate, double dropout){
        this.xLength = length;
        this.yLength = width;
        zLength = isRGB ? 4 : 1;
        this.labelSize = labelSize;
        this.learningRate = learningRate;
        this.isRGB = isRGB;
        this.dropout = dropout;

        cnn = new CNNetwork(xLength, yLength, zLength, labelSize, learningRate, dropout);
    }

    // Train the Classifier
    public void train(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors, Vector<Vector<Double>> testFeatureVectors, int patience, int epochStep, Boolean debug){
        Vector<CNNExample> trainExamples = bulkConvert1Dto3D(trainFeatureVectors);
        Vector<CNNExample> tuneExamples = bulkConvert1Dto3D(tuneFeatureVectors);
        Vector<CNNExample> testExamples = bulkConvert1Dto3D(testFeatureVectors);

        long  overallStart = System.currentTimeMillis(), start = overallStart;
        double bestAcc = cnn.test(tuneExamples, debug);
        int epoch = 0;
        int bestTuneEpoch = 0;

        // Training loop
        for(int i = 0; i < patience; i++){
            if(debug){
                System.out.println("\nEpoch: " + epoch);
            }

            // Train in batch before tuning, if epochStep == 1, then its train once and follow by a tune
            for(int j = 0; j < epochStep; j++){
                Lab.permute(trainExamples);
                cnn.train(trainExamples);
            }

            // Get current accuracy
            if(debug){
                System.out.println("~~~Tuning Set~~~");
            }

            double currAcc = cnn.test(tuneExamples, debug);

            if(currAcc > bestAcc){
                bestAcc = currAcc;
                i = -1;
                bestTuneEpoch = epoch;

                // Keep track of the optimal weights
                cnn.storeOptimalWeights();
            }

            if(debug){
                System.out.println("Done with Epoch # " + Lab.comma(epoch) + ".  Took " + Lab.convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + Lab.convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
            }

            start = System.currentTimeMillis();

            epoch ++;
        }

        cnn.setOptimalWeights();

        System.out.printf("\nBest Tuning Set Accuracy: %.4f%% at Epoch: %d\n", bestAcc * 100, bestTuneEpoch);
    }

    // Test the classifier
    public double test(Vector<Vector<Double>> featureVectors){
        return test(featureVectors, false);
    }

    // Test the classifier
    public double test(Vector<Vector<Double>> featureVectors, Boolean debug){
        Vector<CNNExample> examples = bulkConvert1Dto3D(featureVectors);
        return cnn.test(examples, debug);
    }

    // Classify the example
    public double predict(Vector<Double> example){
        CNNExample cnnexample = convert1Dto3D(example);
        return cnn.predict(cnnexample.example);
    }

    // Utility function to convert feature vectors to 3D array in bulk
    public Vector<CNNExample> bulkConvert1Dto3D(Vector<Vector<Double>> vectors1D){
        Vector<CNNExample> examples = new Vector<CNNExample>(vectors1D.size());

        for(int i = 0; i < vectors1D.size(); i++){
            examples.add(convert1Dto3D(vectors1D.get(i)));
        }

        return examples;
    }

    // Use xLength, yLength and zLength to convert the 1Dvector
    // Utility function to convert feature vectors to 3D array
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