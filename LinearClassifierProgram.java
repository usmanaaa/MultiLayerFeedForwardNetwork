import learn.lc.core.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LinearClassifierProgram {
    public static List<Example> parseFile(String filename) {
        List<Example> examples = new ArrayList<Example>();
        filename = "learn/lc/examples/" + filename;

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] data = line.split(",");
                double[] inputs = new double[data.length - 1];
                for (int i = 0; i < inputs.length; i++) {
                    inputs[i] = Double.parseDouble(data[i]);
                }
                double output = Double.parseDouble(data[data.length - 1]);
                Example example = new Example(inputs, output);
                examples.add(example);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return examples;
    }

    public static void main(String[] args) {
        List<Example> dataset = parseFile(args[0]);
        int nsteps = 700;
        boolean hasAlpha = false;
        double alpha = -1;
        DecayingLearningRateSchedule schedule = new DecayingLearningRateSchedule();

        LinearClassifier lc = new PerceptronClassifier(dataset.get(0).inputs.length);

        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("-n"))
                nsteps = Integer.parseInt(args[i + 1]);
            if (args[i].equals("-a")) {
                alpha = Double.parseDouble(args[i + 1]);
                hasAlpha = true;
            }
            if (args[i].equals("-l")) {
                switch (Integer.parseInt(args[i + 1])) {
                case 1:
                    lc = new PerceptronClassifier(dataset.get(0).inputs.length);
                    System.out.println("Using PerceptronClassifier");
                    break;
                case 2:
                    lc = new LogisticClassifier(dataset.get(0).inputs.length);
                    System.out.println("Using LogisticClassifier");
                    break;
                default:
                    break;
                }
            }
        }

        if (hasAlpha) {
            System.out.println("Using alpha: " + alpha);
            lc.train(dataset, nsteps, alpha, args[0]);

        } else {
            System.out.println("Using decaying rate schedule");
            lc.train(dataset, nsteps, schedule, args[0]);
        }

    }

}
