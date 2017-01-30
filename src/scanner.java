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

        // Iterate through each line in the file.
        int lineCount = 1;
        while(fileScanner.hasNext()) {
            String line = fileScanner.nextLine().trim();

            // Skip blank lines.
            if(line.length() == 0||(new Scanner(line).hasNext()&&new Scanner(line).next().equals("//"))){
                continue;
            }

//            // Print out a message marking each line.
//            System.out.println("Line " + lineCount + "\n=======");
//            lineCount++;

            // Use another scanner to parse each word from the line
            // and print it.
            Scanner lineScanner = new Scanner(line);

//            int wordCount = 1;
            while(lineScanner.hasNext()) {
                String word = lineScanner.next();
//                System.out.println("Word " + wordCount + ": " +
//                        word);
//                wordCount++;
            }

//            System.out.println();
        }
    }


}
