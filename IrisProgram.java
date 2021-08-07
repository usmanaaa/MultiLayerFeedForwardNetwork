import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import learn.nn.core.*;

public class IrisProgram {
    public static List<Example> parseFile() {
        List<Example> examples = new ArrayList<Example>();
        String filename = "learn/nn/examples/iris.data.txt";

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] data = line.split(",");
                double[] inputs = new double[data.length - 1];
                for (int i = 0; i < inputs.length; i++) {
                    inputs[i] = Double.parseDouble(data[i]);
                }
                double[] outputs = new double[] {};
                switch (data[data.length - 1]) {
                case "Iris-setosa":
                    outputs = new double[] { 1, 0, 0 };
                    break;
                case "Iris-versicolor":
                    outputs = new double[] { 0, 1, 0 };
                    break;
                case "Iris-virginica":
                    outputs = new double[] { 0, 0, 1 };
                    break;
                default:
                    break;
                }
                Example example = new Example(inputs, outputs);
                examples.add(example);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return examples;
    }

    public static void main(String[] args) {
        List<Example> dataset = parseFile();
        try {
            File data = new File("reports/iris.data.csv");

            data.createNewFile();

            PrintWriter writer = new PrintWriter(new FileOutputStream(data, false));
            int epochs = 0;
            double alpha = 0.1;

            for (int i = 0; i < args.length; i++) {
                if (args[i].equals("-e"))
                    epochs = Integer.parseInt(args[i + 1]);
                if (args[i].equals("-a")) {
                    alpha = Double.parseDouble(args[i + 1]);
                }

            }

            MultiLayerFeedForwardNeuralNetwork network = new MultiLayerFeedForwardNeuralNetwork(4, 7, 3);
            if (epochs != 0) {
                network.train(dataset, epochs, alpha);
                network.dump();
            } else {
                System.out.println("Please enter epochs!");
            }

            double accuracy = network.kFoldCrossValidate(dataset, 10, epochs, alpha, true);
            System.out.println("Average accuracy: " + accuracy);

            for (int e = 100; e <= 3000; e += 100) {
                double plottedAccuracy = network.kFoldCrossValidate(dataset, 10, e, alpha, false);
                writer.println(plottedAccuracy);

            }
            writer.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }
}
