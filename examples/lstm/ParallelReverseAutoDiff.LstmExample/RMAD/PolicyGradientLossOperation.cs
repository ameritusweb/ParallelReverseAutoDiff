//------------------------------------------------------------------------------
// <copyright file="PolicyGradientLossOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.LstmExample.RMAD
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A loss function for policy gradient optimization.
    /// </summary>
    public class PolicyGradientLossOperation : Operation
    {
        private readonly List<Matrix> chosenActions;
        private readonly int numTimeSteps;
        private readonly double discountFactor;
        private readonly List<double> rewards;

        /// <summary>
        /// Initializes a new instance of the <see cref="PolicyGradientLossOperation"/> class.
        /// </summary>
        /// <param name="chosenActions">The chosen actions of the agent.</param>
        /// <param name="rewards">The positive or negative rewards based on the chosen action.</param>
        /// <param name="numTimeSteps">The number of time steps.</param>
        /// <param name="discountFactor">The discount factor.</param>
        public PolicyGradientLossOperation(List<Matrix> chosenActions, List<double> rewards, int numTimeSteps, double discountFactor)
        {
            this.chosenActions = chosenActions;
            this.numTimeSteps = numTimeSteps;
            this.discountFactor = discountFactor;
            this.rewards = rewards;
        }

        /// <summary>
        /// A factory method for creating a policy gradient loss function.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated policy gradient loss operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new PolicyGradientLossOperation(net.Parameters.ChosenActions, net.Parameters.Rewards, net.Parameters.NumTimeSteps, net.Parameters.DiscountFactor);
        }

        /// <summary>
        /// The backward pass of the policy gradient loss function.
        /// </summary>
        /// <param name="actionProbabilities">The action probabilities.</param>
        /// <returns>The gradient to pass upstream.</returns>
        public override BackwardResult Backward(Matrix actionProbabilities)
        {
            Matrix gradientLossWrtOutput = new Matrix(this.numTimeSteps, 1);
            double beta = 0.01; // Entropy regularization coefficient, adjust as needed
            for (int t = 0; t < this.numTimeSteps; t++)
            {
                double actionProb = actionProbabilities[t][0];

                // Calculate the gradient of the log probability with respect to the output
                double gradLogProbWrtOutput = 0d;
                for (int i = 0; i < this.chosenActions[t].Length; i++)
                {
                    for (int j = 0; j < this.chosenActions[t][i].Length; j++)
                    {
                        if (this.chosenActions[t][i][j] > 0.0d)
                        {
                            gradLogProbWrtOutput += 1.0 / actionProb;
                        }
                    }
                }

                // Multiply the gradient of the log probability by the advantage and negate the result
                double discountedAdvantage = this.rewards[t] * Math.Pow(this.discountFactor, t);
                double gradLossWrtOutput_t = gradLogProbWrtOutput * discountedAdvantage * -1d;

                // Add the gradient of the entropy term
                if ((int)gradLossWrtOutput_t != 0)
                {
                    double gradEntropyWrtOutput = -Math.Log(actionProb) - 1;
                    gradLossWrtOutput_t += beta * gradEntropyWrtOutput;
                }

                gradientLossWrtOutput[t][0] = gradLossWrtOutput_t;
            }

            return new BackwardResultBuilder()
                .AddInputGradient(gradientLossWrtOutput)
                .Build();
        }

        /// <summary>
        /// The forward pass of the policy gradient loss function.
        /// </summary>
        /// <param name="outputsOverTime">The outputs over time.</param>
        /// <returns>The policy gradient loss.</returns>
        public Matrix Forward(Matrix[] outputsOverTime)
        {
            double loss = 0.0;
            double entropy = 0.0;
            double regularizationCoefficient = 0.01; // You can adjust this hyperparameter to control the strength of the regularization

            List<double[]> actionProbabilites = new List<double[]>();

            // Compute the log-probabilities of the chosen actions at each time step
            List<double> logProbs = new List<double>();
            for (int t = 0; t < this.numTimeSteps; t++)
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
                for (int i = 0; i < this.chosenActions[t].Length; i++)
                {
                    for (int j = 0; j < this.chosenActions[t][i].Length; j++)
                    {
                        if (this.chosenActions[t][i][j] > 0.0d)
                        {
                            logProb += Math.Log(actionProbs[0]);
                        }
                    }
                }

                logProbs.Add(logProb);
            }

            // Multiply the log-probabilities by the corresponding discounted rewards and sum over all time steps
            for (int t = 0; t < this.numTimeSteps; t++)
            {
                double discount = Math.Pow(this.discountFactor, t);
                loss += logProbs[t] * this.rewards[t] * discount;
            }

            // Add the entropy regularization term
            loss -= regularizationCoefficient * entropy;

            this.Output = new Matrix(1, 1);
            this.Output[0, 0] = -loss;
            return this.Output;
        }
    }
}
