import java.util.ArrayList;

/**
 * Created by sparsh on 1/29/17.
 */
public class data {
    public Integer  no_of_features;
    public Integer  feature_counter;
    public Integer  no_of_examples;
    public Integer  example_counter;
    public ArrayList features;
    public ArrayList examples;
    public boolean read_features;
    public boolean read_examples;
    public Integer  no_of_labels_possible;
    public String[]   possible_lable_values;

    data(){
        no_of_features   = null;
        feature_counter  =   -1;
        no_of_examples   = null;
        example_counter  =   -1;
        features  =   new ArrayList<features>();
        examples  =   new ArrayList<examples>();
        read_features   =   false;
        read_examples    =   false;
        no_of_labels_possible    =   2;
        possible_lable_values   = new String[no_of_labels_possible];
    }

    public void set_no_of_features(Integer  no_of_features){
        this.no_of_features =   no_of_features;
    }
    public Integer get_no_of_features(){
        return this.no_of_features;
    }
}
