package learn.lc.core;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.List;
import java.util.Random;

import learn.math.util.VectorOps;

abstract public class LinearClassifier {

	public double[] weights;
	Random random = new Random();

	public LinearClassifier(double[] weights) {
		this.weights = weights;
	}

	public LinearClassifier(int ninputs) {
		this.weights = new double[ninputs];
	}

	/**
	 * Update the weights of this LinearClassifer using the given inputs/output
	 * example and learning rate alpha.
	 */
	abstract public void update(double[] x, double y, double alpha);

	/**
	 * Threshold the given value using this LinearClassifier's threshold function.
	 */
	abstract public double threshold(double z);

	/**
	 * Evaluate the given input vector using this LinearClassifier and return the
	 * output value. This value is: Threshold(w \cdot x)
	 */
	public double eval(double[] x) {
		return threshold(VectorOps.dot(this.weights, x));
	}

	/**
	 * Train this LinearClassifier on the given Examples for the given number of
	 * steps, using given learning rate schedule. ``Typically the learning rule is
	 * applied one example at a time, choosing examples at random (as in stochastic
	 * gradient descent).'' See AIMA p. 724.
	 */
	public void train(List<Example> examples, int nsteps, LearningRateSchedule schedule, String filename) {
		try {
			File data = new File("reports/" + filename.substring(0, filename.lastIndexOf('.')) + ".csv");
			data.createNewFile();
			PrintWriter writer = new PrintWriter(new FileOutputStream(data, false));

			int n = examples.size();

			for (int i = 1; i <= nsteps; i++) {
				int j = random.nextInt(n);
				Example ex = examples.get(j);
				this.update(ex.inputs, ex.output, schedule.alpha(i));
				this.trainingReport(examples, i, nsteps, writer);
			}

			writer.close();

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	/**
	 * Train this LinearClassifier on the given Examples for the given number of
	 * steps, using given constant learning rate.
	 */
	public void train(List<Example> examples, int nsteps, double constant_alpha, String filename) {
		train(examples, nsteps, new LearningRateSchedule() {
			public double alpha(int t) {
				return constant_alpha;
			}
		}, filename);
	}

	/**
	 * This method is called after each weight update during training. Subclasses
	 * can override it to gather statistics or update displays.
	 */
	protected void trainingReport(List<Example> examples, int stepnum, int nsteps, PrintWriter writer) {
		writer.println(stepnum + ", " + accuracy(examples));

	}

	/**
	 * Return the squared error per example for this Linearlassifier using the L2
	 * (squared error) loss function.
	 */
	public double squaredErrorPerSample(List<Example> examples) {
		double sum = 0.0;
		for (Example ex : examples) {
			double result = eval(ex.inputs);
			double error = ex.output - result;
			sum += error * error;
		}
		return sum / examples.size();
	}

	/**
	 * Return the proportion of the given Examples that are classified correctly by
	 * this LinearClassifier. This is probably only meaningful for classifiers that
	 * use a hard threshold. Use with care.
	 */
	public double accuracy(List<Example> examples) {
		int ncorrect = 0;
		for (Example ex : examples) {
			double result = eval(ex.inputs);
			if (result == ex.output) {
				ncorrect += 1;
			}
		}
		return (double) ncorrect / examples.size();
	}

}
