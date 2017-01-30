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

    public data data;

	public data read_files(String args[]) {
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

			// Skip if blank lines or comments.
			if(line.length() == 0||(new Scanner(line).hasNext()&&new Scanner(line).next().equals("//"))){
				continue;
			}

			//if first valid line and no_of_features is empty then incoming line has no_of_features
			if(data.get_no_of_features()==null){
				data.set_no_of_features(Integer.parseInt(line.trim()));
				continue;
			}
			//check if the incoming line is a feature then increment feature count
			if(data.no_of_features!=null&&data.feature_counter<data.no_of_features-1){
				data.feature_counter++;
				data.read_features   =   true;
			}
			else{
				data.read_features   =   false;
			}

			//read features
			if (data.read_features) {
				String line_components[]   =   line.split("-");
				features feature =  new features();
				feature.name  =   line_components[0].trim();
				feature.values    =line_components[1].trim().split(" ");
				data.features.add(feature);
				continue;
			}

			//after reading features, read possible values of labels
			if(data.no_of_labels_possible>0){
				data.possible_lable_values[2-data.no_of_labels_possible]   = line.trim();
				data.no_of_labels_possible--;
				continue;
			}

			//after reading features take the line containing no_of_examples
			if(data.no_of_examples==null){
				data.no_of_examples = Integer.parseInt(line.trim());
				continue;
			}

			//check if the incoming line is a example then increment example_count
			if(data.no_of_examples!=null&&data.example_counter<data.no_of_examples){
				data.example_counter++;
				data.read_examples   =   true;
			}
			else{
				data.read_examples   =   false;
			}


			//read examples
			if (data.read_examples) {
				String line_components[]   =   line.split(" ");
				examples example    =   new examples();
				example.name  =   line_components[0];
                String new_line_components[] =   Arrays.copyOfRange(line_components, 1, line_components.length);
				example.feature_values  = new_line_components;
				data.examples.add(example);
			}
		}
		return data;
	}


}
