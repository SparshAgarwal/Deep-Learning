import java.util.Arrays;

/**
 * Created by sparsh on 1/30/2017.
 */
public class features {
    public String name;
    public String[] values;

    @Override
    public String toString(){
        return  "Name:" +   name    +   "values:"   + Arrays.toString(values);
    }
}
