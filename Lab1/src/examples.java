import java.util.ArrayList;

class examples {
    public String name;
    public ArrayList<String> feature_values;

    examples(){
        feature_values  =   new ArrayList<String>();
    }

    @Override
    public String toString(){
        return  "Name:" +   name    +   "values:"   + feature_values;
    }
}