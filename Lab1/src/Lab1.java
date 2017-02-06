/**
 * Created by sparsh on 1/30/2017.
 */
public class Lab1 {
    public static void main(String args[]){
        scanner S = new scanner();
        data train  =   S.read_files(args);
        System.out.print(train.features.get(0));
    }
}
