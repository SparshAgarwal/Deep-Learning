/**
 * ScannerSample.java
 * September 2014
 * Sample program for Professor Shavlik's CS 540 class which demonstrates how
 * to read a file using a Scanner.
 *
 * If you are running this in Eclipse, you will need to add the filename as a
 * command-line argument. Go to  Project > Properties > Run/Debug Settings >
 * New... and create a new Run Configuration for a Java Application. Click on
 * the Arguments tab and type the name of the file you want to read in. If you
 * ever want to change this, return to the same menu and click the Edit...
 * button.
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class scanner {

    public int no_of_features   = Integer.parseInt(null);
    public int feature_counter  =   -1;
    public int no_of_objects   = Integer.parseInt(null);
    public int object_counter   =   -1;
    public features[] features;
    public objects[]    objects;
    boolean read_features   =   false;
    boolean read_objects    =   false;

    public void read_files(String args[]) {
        // Make sure an input file was specified on the command line.
        // If this fails and you are using Eclipse, see the comments
        // in the header above.
        if (args.length != 1) {
            System.err.println("Please supply a filename on the " +
                    "command line: java ScannerSample" +
                    " <filename>");
            System.exit(1);
        }

        // Try creating a scanner to read the input file.
        Scanner fileScanner = null;
        try {
            fileScanner = new Scanner(new File(args[0]));
        } catch(FileNotFoundException e) {
            System.err.println("Could not find file '" + args[0] +
                    "'.");
            System.exit(1);
        }

        // Iterate through each line in the file
        while(fileScanner.hasNext()) {
            String line = fileScanner.nextLine().trim();

            // Skip blank lines and comments.
            if(line.length() == 0||(new Scanner(line).hasNext()&&new Scanner(line).next().equals("//"))){
                continue;
            }

            //create line scanner
            Scanner lineScanner = new Scanner(line);

            //check if the incoming line is a feature and increment feature count
            if(no_of_features!=Integer.parseInt(null)&&feature_counter<no_of_features){
                feature_counter++;
            }
            else{
                read_features   =   false;
            }

                //if first valid line and no_of_features is empty then incoming line has no_of_features
                if(no_of_features==Integer.parseInt(null)){
                    no_of_features = Integer.parseInt(lineScanner.next());
                    features    =   new features[no_of_features];
                    continue;
                }
                //read features
                if (read_features) {
                    String feature_line =   lineScanner.next().split("-");
                    features[feature_counter].name  =   lineScanner.next();
                    features[feature_counter].values    +=lineScanner.next();

                }


            if(feature_counter<no_of_features)
            {
                if()

                feature_counter++;
            }

        }
    }


}
