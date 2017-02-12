/**
 * Created by sparsh on 1/28/17.
 */
import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

public class Lab1 {
    public static void main(String args[]){
        scanner train = new scanner();
        train.read_files(args[0]);
        scanner tune = new scanner();
        tune.read_files(args[1]);
        scanner test = new scanner();
        test.read_files(args[2]);
        scanner req_train    =   map_data(train);
        scanner req_tune   =   map_data(tune);
        scanner req_test    =   map_data(test);
        ArrayList<Double> best_weight = new ArrayList<Double>();
        Double cases_correct = 0.0;
        Double final_accuracy;

        best_weight =   perceptron(req_train,req_tune);

        for(int i=0;i<req_test.no_of_examples;i++){
            examples curr_test_example  =   req_test.req_examples.get(i);
            Double curr_inp = 0.0;
            Double threshold    =   0.0;
            Double outcome;
            for(int j=1;j<req_test.no_of_features;j++){
                curr_inp +=   best_weight.get(j-1) * Double.parseDouble(curr_test_example.feature_values.get(j));
            }
            if (curr_inp > threshold) {
                outcome = 1.0;
                String category = "lowToMid";
                System.out.println("Predicted category for "+test.req_examples.get(i).name+" : "+category);
            } else {
                outcome = -1.0;
                String category = "midToHigh";
                System.out.println("Predicted category for "+test.req_examples.get(i).name+" : "+category);
            }
            if(Double.parseDouble(curr_test_example.feature_values.get(0))==outcome){
                cases_correct++;
            }
        }
        final_accuracy    =   cases_correct/req_test.no_of_examples;
        System.out.print("Final accuracy over test set is : " + 100*final_accuracy +"%");
    }

    public static ArrayList<Double> perceptron(scanner train, scanner tune) {
        ArrayList<Double> weights = new ArrayList<Double>(Collections.<Double>nCopies(train.no_of_features+1, 0.0));
        Double threshold   =   0.0;
        int outcome =   0;
        Double inp=  0.0;
        double learning_rate    =  0.1;
        double  diff;
        Double test_accracy =  -1.0;
        Double obt_accuracy = 0.0;
        Double cases_correct   =   0.0;

        while(test_accracy<obt_accuracy) {
            test_accracy = obt_accuracy;
            for (int epoch = 0; epoch < 30; epoch++) {
                for (int i = 0; i < train.no_of_examples; i++) {
                    examples curr_example = train.req_examples.get(i);
                    for (int j = 1; j < train.no_of_features; j++) {
                        inp += weights.get(j-1) * Double.parseDouble(curr_example.feature_values.get(j));
                    }
                    if (inp > threshold) {
                        outcome = 1;
                    } else {
                        outcome = -1;
                    }
                    diff = learning_rate * (Double.parseDouble(curr_example.feature_values.get(0)) - outcome);
                    for (int j = 1; j <= weights.size(); j++) {
                        weights.set(j-1, weights.get(j-1) + (diff * Double.parseDouble(curr_example.feature_values.get(j))));
                    }
                }
            }
            for(int i=0;i<tune.no_of_examples;i++){
                examples curr_tune_example  =   tune.req_examples.get(i);
                Double curr_inp = 0.0;

                for(int j=1;j<tune.no_of_features;j++){
                    curr_inp +=   weights.get(j-1) * Double.parseDouble(curr_tune_example.feature_values.get(j));
                }
                if (curr_inp > threshold) {
                    outcome = 1;
                } else {
                    outcome = -1;
                }
                if(Double.parseDouble(curr_tune_example.feature_values.get(0))==outcome){
                    cases_correct++;
                }
            }
            obt_accuracy    =   cases_correct/tune.no_of_examples;
            cases_correct   =   0.0;
        }
        return weights;
    }
    


    public static scanner map_data(scanner data){
        scanner ret_data    =   new scanner();
        int i   =   0;
        ret_data.no_of_examples = data.no_of_examples;
        ret_data.no_of_features =   data.no_of_features;

        while(i<ret_data.no_of_examples){
            examples first_example   =   (examples) data.req_examples.get(i);
            examples req_example = new examples();
            req_example.name    =   Integer.toString(i);
            if(first_example.feature_values.get(0).equals("lowToMid")){
                req_example.feature_values.add("1");
            }
            else {
                req_example.feature_values.add("-1");
            }
            for(int j=1; j<first_example.feature_values.size();j++){
                if(first_example.feature_values.get(j).equals("T")){
                    req_example.feature_values.add("1");
                }
                else {
                    req_example.feature_values.add("-1");
                }
            }
            req_example.feature_values.add("1");//for bias
            ret_data.req_examples.add(req_example);
            i+=1;
        }
        return ret_data;
    }
}



