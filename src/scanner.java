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

import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class scanner {

    public Integer  no_of_features = null;;
    public Integer  feature_counter  =   -1;
    public Integer  no_of_examples = null;
    public Integer  example_counter  =   -1;
    public ArrayList features  =   new ArrayList<features>();
    public ArrayList examples  =   new ArrayList<examples>();
    public boolean read_features   =   false;
    public boolean read_examples    =   false;
    public Integer  no_of_labels_possible    =   2;
    public String[]   possible_lable_values   = new String[no_of_labels_possible];

    
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
			if(line.length() == 0||(new Scanner(line).hasNext()&&new Scanner(line).next().equals("//"))){
				continue;
			}

			//if first valid line and no_of_features is empty then incoming line has no_of_features
			if(no_of_features==null){
				no_of_features  =   Integer.parseInt(line.trim());
				continue;
			}
			//check if the incoming line is a feature then increment feature count
			if(no_of_features!=null&&feature_counter<no_of_features-1){
				feature_counter++;
				read_features   =   true;
			}
			else{
				read_features   =   false;
			}

			//read features
			if (read_features) {
				String line_components[]   =   line.split("-");
				features feature =  new features();
				feature.name  =   line_components[0].trim();
				feature.values    =line_components[1].trim().split(" ");
				features.add(feature);
				continue;
			}

			//after reading features, read possible values of labels
			if(no_of_labels_possible>0){
				possible_lable_values[2-no_of_labels_possible]   = line.trim();
				no_of_labels_possible--;
				continue;
			}

			//after reading features take the line containing no_of_examples
			if(no_of_examples==null){
				no_of_examples = Integer.parseInt(line.trim());
				continue;
			}

			//check if the incoming line is a example then increment example_count
			if(no_of_examples!=null&&example_counter<no_of_examples){
				example_counter++;
				read_examples   =   true;
			}
			else{
				read_examples   =   false;
			}


			//read examples
			if (read_examples) {
				String line_components[]   =   line.split(" ");
				examples example    =   new examples();
				example.name  =   line_components[0];
                String new_line_components[] =   Arrays.copyOfRange(line_components, 1, line_components.length);
				example.feature_values  = new_line_components;
				examples.add(example);
			}
		}
	}


}
