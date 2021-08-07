package learn.lc.core;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.List;

public class PerceptronClassifier extends LinearClassifier {

	public PerceptronClassifier(double[] weights) {
		super(weights);
	}

	public PerceptronClassifier(int ninputs) {
		super(ninputs);
	}

	/**
	 * A PerceptronClassifier uses the perceptron learning rule (AIMA Eq. 18.7): w_i
	 * \leftarrow w_i+\alpha(y-h_w(x)) \times x_i
	 */
	public void update(double[] x, double y, double alpha) {
		for (int i = 0; i < this.weights.length; i++) {
			this.weights[i] += (alpha * (y - eval(x))) * x[i];
		}
	}

	/**
	 * A PerceptronClassifier uses a hard 0/1 threshold.
	 */
	public double threshold(double z) {
		return z >= 0 ? 1 : 0;
	}

}
