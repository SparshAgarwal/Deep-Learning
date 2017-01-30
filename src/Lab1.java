/**
 * Created by sparsh on 1/30/2017.
 */
public class Lab1 {
    public static void main(String args[]){
        scanner train = new scanner();
        train.read_files(args[0]);
        System.out.print(train.examples.get(2));
    }
}
