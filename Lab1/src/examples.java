import java.util.Arrays;

/**
 * Created by sparsh on 1/30/2017.
 */
public class examples {
    public String name;
    public String[] feature_values;

    @Override
    public String toString(){
        return  "Name:" +   name    +   "values:"   + Arrays.toString(feature_values);
    }
}
