package learn.nn.core;

/**
 * A LogisticUnit is a Unit that uses a sigmoid activation function.
 */
public class LogisticUnit extends NeuronUnit {

	/**
	 * The activation function for a LogisticUnit is a 0-1 sigmoid centered at z=0:
	 * 1/(1+e^(-z)). (AIMA Fig 18.7)
	 */
	@Override
	public double activation(double z) {
		return 1 / (1 + Math.exp(-z));
	}

	/**
	 * Derivative of the activation function for a LogisticUnit. For
	 * g(z)=1/(1+e^(-z)), g'(z)=g(z)*(1-g(z)) (AIMA p. 727).
	 * 
	 * @see https://calculus.subwiki.org/wiki/Logistic_function#First_derivative
	 */
	public double activationPrime(double z) {
		double y = activation(z);
		return y * (1.0 - y);
	}

	//// EXPERIMENT WITH THE LOOP SIZE
	/**
	 * Update this unit's weights using the logistic regression gradient descent
	 * learning rule (AIMA Eq 18.8). Remember: If there are n input attributes in
	 * vector x, then there are n+1 weights including the bias weight w_0.
	 */
	@Override
	public void update(double[] x, double y, double alpha) {
		for (int i = 0; i < this.incomingConnections.size(); i++) {
			double wi = this.getWeight(i);
			wi += (alpha * (y - h_w(x))) * (h_w(x) * (1 - h_w(x))) * x[i];
			this.setWeight(i, wi);
		}
	}

}
