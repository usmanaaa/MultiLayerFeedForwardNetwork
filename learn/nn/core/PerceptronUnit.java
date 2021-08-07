package learn.nn.core;

/**
 * A PerceptronUnit is a Unit that uses a hard threshold activation function.
 */
public class PerceptronUnit extends NeuronUnit {

	/**
	 * The activation function for a Perceptron is a hard 0/1 threshold at z=0.
	 * (AIMA Fig 18.7)
	 */
	@Override
	public double activation(double z) {
		return z >= 0 ? 1 : 0;
	}

	/**
	 * Update this unit's weights using the Perceptron learning rule (AIMA Eq 18.7).
	 * Remember: If there are n input attributes in vector x, then there are n+1
	 * weights including the bias weight w_0.
	 */
	@Override
	public void update(double[] x, double y, double alpha) {
		for (int i = 0; i < this.incomingConnections.size(); i++) {
			double wi = this.getWeight(i);
			wi += (alpha * (y - h_w(x))) * x[i];
			this.setWeight(i, wi);
		}
	}
}
