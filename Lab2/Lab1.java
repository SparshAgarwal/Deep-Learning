/**
 * Created by sparsh on 1/28/17.
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class Lab1 {
    public static void main(String args[]){
        scanner file = new scanner();
        file.read_files(args[0]);
//        scanner req_train    =   map_data(file);
//        ArrayList<Double> best_weight = new ArrayList<Double>();
//        Double cases_correct = 0.0;
//        Double final_accuracy;
//
//        best_weight =   perceptron(req_train,req_tune);
//
//        for(int i=0;i<req_test.no_of_examples;i++){
//            examples curr_test_example  =   req_test.req_examples.get(i);
//            Double curr_inp = 0.0;
//            Double threshold    =   0.0;
//            Double outcome;
//            for(int j=1;j<req_test.no_of_features;j++){
//                curr_inp +=   best_weight.get(j-1) * Double.parseDouble(curr_test_example.feature_values.get(j));
//            }
//            if (curr_inp > threshold) {
//                outcome = 1.0;
//                String category = "lowToMid";
//                System.out.println("Predicted category for "+test.req_examples.get(i).name+" : "+category);
//            } else {
//                outcome = -1.0;
//                String category = "midToHigh";
//                System.out.println("Predicted category for "+test.req_examples.get(i).name+" : "+category);
//            }
//            if(Double.parseDouble(curr_test_example.feature_values.get(0))==outcome){
//                cases_correct++;
//            }
//        }
//        final_accuracy    =   cases_correct/req_test.no_of_examples;
//        System.out.print("Final accuracy over test set is : " + 100*final_accuracy +"%");
    }

//    public static ArrayList<Double> perceptron(scanner train, scanner tune) {
//        ArrayList<Double> weights = new ArrayList<Double>(Collections.<Double>nCopies(train.no_of_features+1, 0.0));
//        Double threshold   =   0.0;
//        int outcome =   0;
//        Double inp=  0.0;
//        double learning_rate    =  0.1;
//        double  diff;
//        Double test_accracy =  -1.0;
//        Double obt_accuracy = 0.0;
//        Double cases_correct   =   0.0;
//
//        while(test_accracy<obt_accuracy) {
//            test_accracy = obt_accuracy;
//            for (int epoch = 0; epoch < 30; epoch++) {
//                for (int i = 0; i < train.no_of_examples; i++) {
//                    examples curr_example = train.req_examples.get(i);
//                    for (int j = 1; j < train.no_of_features; j++) {
//                        inp += weights.get(j-1) * Double.parseDouble(curr_example.feature_values.get(j));
//                    }
//                    if (inp > threshold) {
//                        outcome = 1;
//                    } else {
//                        outcome = -1;
//                    }
//                    diff = learning_rate * (Double.parseDouble(curr_example.feature_values.get(0)) - outcome);
//                    for (int j = 1; j <= weights.size(); j++) {
//                        weights.set(j-1, weights.get(j-1) + (diff * Double.parseDouble(curr_example.feature_values.get(j))));
//                    }
//                }
//            }
//            for(int i=0;i<tune.no_of_examples;i++){
//                examples curr_tune_example  =   tune.req_examples.get(i);
//                Double curr_inp = 0.0;
//
//                for(int j=1;j<tune.no_of_features;j++){
//                    curr_inp +=   weights.get(j-1) * Double.parseDouble(curr_tune_example.feature_values.get(j));
//                }
//                if (curr_inp > threshold) {
//                    outcome = 1;
//                } else {
//                    outcome = -1;
//                }
//                if(Double.parseDouble(curr_tune_example.feature_values.get(0))==outcome){
//                    cases_correct++;
//                }
//            }
//            obt_accuracy    =   cases_correct/tune.no_of_examples;
//            cases_correct   =   0.0;
//        }
//        return weights;
//    }
    

//
//    public static scanner map_data(scanner data){
//        scanner ret_data    =   new scanner();
//        int i   =   0;
//        ret_data.no_of_examples = data.no_of_examples;
//        ret_data.no_of_features =   data.no_of_features;
//
//        while(i<ret_data.no_of_examples){
//            examples first_example   =   (examples) data.req_examples.get(i);
//            examples req_example = new examples();
//            req_example.name    =   Integer.toString(i);
//            if(first_example.feature_values.get(0).equals("lowToMid")){
//                req_example.feature_values.add("1");
//            }
//            else {
//                req_example.feature_values.add("-1");
//            }
//            for(int j=1; j<first_example.feature_values.size();j++){
//                if(first_example.feature_values.get(j).equals("T")){
//                    req_example.feature_values.add("1");
//                }
//                else {
//                    req_example.feature_values.add("-1");
//                }
//            }
//            req_example.feature_values.add("1");//for bias
//            ret_data.req_examples.add(req_example);
//            i+=1;
//        }
//        return ret_data;
//    }
}

class scanner {

    public Integer  protien_number = 0;
    public Integer  acid_number = 0;
    public ArrayList<List<String>> amino_acids  =   new ArrayList<List<String>>();
    public ArrayList<List<String>> protien_label  =   new ArrayList<List<String>>();
    public List<String> amino = new ArrayList<String>();
    public List<String> label = new ArrayList<String>();


    public void read_files(String args) {
        // Make sure an input file was specified on the command line.
        // If this fails and you are using Eclipse, see the comments
        // in the header above.
        if (args == null) {
            System.err.println("Please supply a filename on the " +
                    "command line: java ScannerSample" +
                    " <filename>");
            System.exit(1);
        }

        // Try creating a scanner to read the input file.
        Scanner fileScanner = null;
        try {
            fileScanner = new Scanner(new File(args));
        } catch(FileNotFoundException e) {
            System.err.println("Could not find file '" + args +
                    "'.");
            System.exit(1);
        }

        // Iterate through each line in the file
        while(fileScanner.hasNext()) {
            String line = fileScanner.nextLine().trim();

            // Skip if blank lines or comments.
            if(line.length() == 0||(new Scanner(line).hasNext()&&new Scanner(line).next().equals("//"))||
                    (new Scanner(line).hasNext()&&new Scanner(line).next().equals("#"))){
                continue;
            }
            while((fileScanner.hasNext())&&(line.contains("<>")||line.contains("end")||line.contains("<end>"))){
                line = fileScanner.nextLine().trim();
            }
            protien_number++;
            while((!(line.contains("<>")||line.contains("end")||line.contains("<end>")))&&(fileScanner.hasNext())) {
                if(line.length() == 0||(new Scanner(line).hasNext()&&new Scanner(line).next().equals("//"))||
                        (new Scanner(line).hasNext()&&new Scanner(line).next().equals("#"))){
                    continue;
                }
                String line_components[] = line.split(" ");
                amino.add(line_components[0]);
                label.add(line_components[1]);
                acid_number++;
                line = fileScanner.nextLine().trim();
            }
            amino_acids.add(amino);
            protien_label.add(label);
        }
        System.out.print(protien_number);
    }


}


