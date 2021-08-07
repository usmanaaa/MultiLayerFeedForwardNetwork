package learn.lc.core;

import java.io.PrintWriter;
import java.util.List;

public class LogisticClassifier extends LinearClassifier {

	public LogisticClassifier(double[] weights) {
		super(weights);
	}

	public LogisticClassifier(int ninputs) {
		super(ninputs);
	}

	/**
	 * A LogisticClassifier uses the logistic update rule (AIMA Eq. 18.8): w_i
	 * \leftarrow w_i+\alpha(y-h_w(x)) \times h_w(x)(1-h_w(x)) \times x_i
	 */
	public void update(double[] x, double y, double alpha) {
		for (int i = 0; i < this.weights.length; i++) {
			this.weights[i] += (alpha * (y - eval(x))) * (eval(x) * (1 - eval(x))) * x[i];
		}

	}

	/**
	 * A LogisticClassifier uses a 0/1 sigmoid threshold at z=0.
	 */
	public double threshold(double z) {
		return 1 / (1 + Math.exp(-z));
	}

	/**
	 * This method is called after each weight update during training. Subclasses
	 * can override it to gather statistics or update displays.
	 */
	@Override
	protected void trainingReport(List<Example> examples, int stepnum, int nsteps, PrintWriter writer) {
		writer.println(stepnum + ", " + (1 - squaredErrorPerSample(examples)));

	}
}
