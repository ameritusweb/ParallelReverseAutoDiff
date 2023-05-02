namespace ParallelReverseAutoDiff.LstmExample.RMAD
{
    using ParallelReverseAutoDiff.RMAD;

    public class PolicyGradientLossOperation : Operation
    {
        private List<double[][]> _chosenActions;
        private int _numTimeSteps;
        private double _discountFactor;
        private List<double> _rewards;

        public PolicyGradientLossOperation(List<double[][]> chosenActions, List<double> rewards, int numTimeSteps, double discountFactor) : base()
        {
            _chosenActions = chosenActions;
            _numTimeSteps = numTimeSteps;
            _discountFactor = discountFactor;
            _rewards = rewards;
        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new PolicyGradientLossOperation(net.GetChosenActions(), net.GetRewards(), net.GetNumTimeSteps(), net.GetDiscountFactor());
        }

        public override (double[][]?, double[][]?) Backward(double[][] actionProbabilities)
        {
            double[][] gradientLossWrtOutput = new double[_numTimeSteps][];
            double beta = 0.01; // Entropy regularization coefficient, adjust as needed
            for (int t = 0; t < _numTimeSteps; t++)
            {
                gradientLossWrtOutput[t] = new double[1];
                double actionProb = actionProbabilities[t][0];

                // Calculate the gradient of the log probability with respect to the output
                double gradLogProbWrtOutput = 0d;
                for (int i = 0; i < _chosenActions[t].Length; i++)
                {
                    for (int j = 0; j < _chosenActions[t][i].Length; j++)
                    {
                        if (_chosenActions[t][i][j] > 0.0d)
                        {
                            gradLogProbWrtOutput += (1.0 / actionProb);
                        }
                    }
                }

                // Multiply the gradient of the log probability by the advantage and negate the result
                double discountedAdvantage = _rewards[t] * Math.Pow(_discountFactor, t);
                double gradLossWrtOutput_t = gradLogProbWrtOutput * discountedAdvantage * -1d;

                // Add the gradient of the entropy term
                if (gradLossWrtOutput_t != 0.0d)
                {
                    double gradEntropyWrtOutput = -Math.Log(actionProb) - 1;
                    gradLossWrtOutput_t += beta * gradEntropyWrtOutput;
                }

                gradientLossWrtOutput[t][0] = gradLossWrtOutput_t;
            }

            return (gradientLossWrtOutput, gradientLossWrtOutput);
        }

        public double[][] Forward(double[][][] outputsOverTime)
        {
            double loss = 0.0;
            double entropy = 0.0;
            double regularizationCoefficient = 0.01; // You can adjust this hyperparameter to control the strength of the regularization

            List<double[]> actionProbabilites = new List<double[]>();

            // Compute the log-probabilities of the chosen actions at each time step
            List<double> logProbs = new List<double>();
            for (int t = 0; t < _numTimeSteps; t++)
            {
                double[] actionProbs = outputsOverTime[t][0];
                actionProbabilites.Add(actionProbs);

                // Compute the entropy at this time step
                double timeStepEntropy = 0.0;
                for (int i = 0; i < actionProbs.Length; i++)
                {
                    timeStepEntropy -= actionProbs[i] * Math.Log(actionProbs[i]);
                }
                entropy += timeStepEntropy;

                // Iterate over the input and output positions on the chessboard
                double logProb = 0.0;
                for (int i = 0; i < _chosenActions[t].Length; i++)
                {
                    for (int j = 0; j < _chosenActions[t][i].Length; j++)
                    {
                        if (_chosenActions[t][i][j] > 0.0d)
                        {
                            logProb += Math.Log(actionProbs[0]);
                        }
                    }
                }
                logProbs.Add(logProb);
            }

            // Multiply the log-probabilities by the corresponding discounted rewards and sum over all time steps
            for (int t = 0; t < _numTimeSteps; t++)
            {
                double discount = Math.Pow(_discountFactor, t);
                loss += logProbs[t] * _rewards[t] * discount;
            }

            // Add the entropy regularization term
            loss -= regularizationCoefficient * entropy;

            _output = new double[][] { new double[] { -loss } };
            return _output;
        }
    }
}
