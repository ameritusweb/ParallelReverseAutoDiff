//------------------------------------------------------------------------------
// <copyright file="CategoricalVarianceBinaryLossOperation.cs" author="ameritusweb" date="9/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FsmnnExample.FiniteStateMachine.RMAD
{
    using System;
    using ParallelReverseAutoDiff.GnnExample.Common;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Categorical variance binary loss operation.
    /// Uses an algothim to determine the optimal threshold.
    /// Utilizes early stopping in the loss function and subsequent gradient calculation.
    /// </summary>
    public class CategoricalVarianceBinaryLossOperation
    {
        private const double Lambda = 50d;
        private const double Alpha = 1d;
        private const double Epsilon = 1E-9d;
        private Matrix predicted;
        private Matrix targetMatrix;
        private int targetIndex;
        private double trueVariance;
        private double mean;
        private double variance;

        /// <summary>
        /// Performs the forward operation.
        /// </summary>
        /// <param name="predicted">1xN matrix of predicted probabilities.</param>
        /// <param name="targetMatrix">The target matrix of ones or zeroes.</param>
        /// <param name="trueVariance">1xN matrix of true labels (one-hot encoded).</param>
        /// <returns>The computed loss.</returns>
        public double Forward(Matrix predicted, Matrix targetMatrix, double trueVariance)
        {
            this.predicted = predicted;
            this.targetMatrix = targetMatrix;
            this.trueVariance = trueVariance;
            this.targetIndex = -1;

            double minLoss = double.MaxValue;
            double optimalThreshold = 0;

            // Sort predicted values
            double[] sortedPredicted = predicted[0].OrderByDescending(x => x).ToArray();

            // Calculate differences
            double[] differences = new double[sortedPredicted.Length - 1];
            for (int i = 0; i < sortedPredicted.Length - 1; i++)
            {
                differences[i] = sortedPredicted[i] - sortedPredicted[i + 1];
            }

            // Add end-points and create dynamic thresholds
            List<double> dynamicThresholds = new List<double> { sortedPredicted[0] + Epsilon };  // Starting just above the max value
            foreach (double diff in differences)
            {
                dynamicThresholds.Add(dynamicThresholds.Last() - diff);
            }

            dynamicThresholds.Add(0);  // Ending at 0

            // Search for optimal threshold using dynamic step sizes
            foreach (double t in dynamicThresholds)
            {
                double currentLoss = 0;

                for (int i = 0; i < predicted.Cols; i++)
                {
                    double binary = predicted[0][i] > t ? 1d : 0d;
                    currentLoss += Math.Abs(binary - targetMatrix[0][i]);  // L1 loss as an example
                }

                if (currentLoss < minLoss)
                {
                    minLoss = currentLoss;
                    optimalThreshold = t;
                }
            }

            double mean = predicted[0].Average();
            this.mean = mean;

            double variance = predicted[0].Select(x => (x - mean) * (x - mean)).Average();
            this.variance = variance;

            var threshold = optimalThreshold;  // Use the algorithmically determined optimal threshold

            double maxDiff = 0d;
            foreach (var (item, index) in predicted[0].WithIndex())
            {
                var binary = item > threshold ? 1d : 0d;
                if (binary != targetMatrix[0][index])
                {
                    // item is currently zero, so needs to move up
                    if (binary == 0d)
                    {
                        if (Math.Abs(item - threshold) > maxDiff)
                        {
                            this.targetIndex = index;
                            maxDiff = Math.Abs(item - threshold);
                        }
                    }
                }
            }

            double loss = Math.Pow(variance - trueVariance, 2);

            return loss;
        }

        /// <summary>
        /// Computes the gradient with respect to the predicted probabilities.
        /// </summary>
        /// <returns>1xN matrix of gradients.</returns>
        public Matrix Backward()
        {
            Matrix dOutput = new Matrix(1, this.predicted.Cols);

            // Compute the gradient with respect to each element of the input vector
            for (int i = 0; i < this.predicted.Cols; i++)
            {
                dOutput[0][i] = (Lambda * ((4.0 / this.predicted.Cols) *
                    (this.variance - this.trueVariance) *
                    (this.predicted[0][i] - this.mean))) + (Alpha * (this.targetIndex == i ? -1 / (this.predicted[0][i] + Epsilon) : 0d));
            }

            return dOutput;
        }
    }
}
