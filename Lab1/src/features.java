import java.util.ArrayList;

class features {
    public String name;
    public ArrayList<String> values;

    features(){
        values    =   new ArrayList<String>();
    }

    @Override
    public String toString(){
        return  "Name:" +   name    +   "values:"   + values;
    }
}