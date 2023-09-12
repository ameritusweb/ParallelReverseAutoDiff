//------------------------------------------------------------------------------
// <copyright file="CategoricalVarianceBinaryThresholdLossOperation.cs" author="ameritusweb" date="9/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FsmnnExample.FiniteStateMachine.RMAD
{
    using System;
    using ParallelReverseAutoDiff.GnnExample.Common;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Categorical variance binary threshold loss operation.
    /// Uses an algothim to determine the optimal threshold.
    /// Utilizes early stopping in the loss function and subsequent gradient calculation.
    /// </summary>
    public class CategoricalVarianceBinaryThresholdLossOperation
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
            // Initialize member variables
            this.predicted = predicted;
            this.targetMatrix = targetMatrix;
            this.trueVariance = trueVariance;
            this.targetIndex = -1;

            // Variables to keep track of minimum loss and optimal threshold
            double minLoss = double.MaxValue;
            double optimalThreshold = 0;

            // Sort the predicted probabilities in descending order
            double[] sortedPredicted = predicted[0].OrderByDescending(x => x).ToArray();

            // Compute the differences between adjacent sorted predicted probabilities
            double[] differences = new double[sortedPredicted.Length - 1];
            for (int i = 0; i < sortedPredicted.Length - 1; i++)
            {
                differences[i] = sortedPredicted[i] - sortedPredicted[i + 1];
            }

            // Create a list of dynamic thresholds based on sorted predicted probabilities
            List<double> dynamicThresholds = new List<double> { sortedPredicted[0] + Epsilon };  // Start just above the max value
            foreach (double diff in differences)
            {
                dynamicThresholds.Add(dynamicThresholds.Last() - diff);
            }

            dynamicThresholds.Add(0);  // End at 0

            double gap = 0d;

            // Search for the optimal threshold using dynamic step sizes
            foreach (var (t, index) in dynamicThresholds.WithIndex())
            {
                double currentLoss = 0;
                for (int i = 0; i < predicted.Cols; i++)
                {
                    double binary = predicted[0][i] > t ? 1d : 0d;
                    currentLoss += Math.Abs(binary - targetMatrix[0][i]);  // L1 loss
                }

                // Update optimal threshold if a lower loss is found
                if (currentLoss < minLoss)
                {
                    minLoss = currentLoss;
                    optimalThreshold = t;
                    if (index == 0)
                    {
                        gap = Math.Abs(optimalThreshold - dynamicThresholds[index + 1]);
                    }
                    else if (index == dynamicThresholds.Count)
                    {
                        gap = Math.Abs(optimalThreshold - dynamicThresholds[index - 1]);
                    }
                    else
                    {
                        gap = Math.Max(Math.Abs(optimalThreshold - dynamicThresholds[index - 1]), Math.Abs(optimalThreshold - dynamicThresholds[index + 1]));
                    }
                }
            }

            // Compute the mean and variance of the predicted probabilities
            double mean = predicted[0].Average();
            this.mean = mean;
            double variance = predicted[0].Select(x => (x - mean) * (x - mean)).Average();
            this.variance = variance;
            Console.WriteLine("Variance: " + variance);

            // Use algorithmically determined optimal threshold
            var threshold = optimalThreshold;

            // Initialize variables for edge case handling
            double maxDiff = 0d;
            double closestOne = double.MaxValue;
            int closestIndex = -1;

            // Iterate through each predicted value and its index
            foreach (var (item, index) in predicted[0].WithIndex())
            {
                var binary = item > threshold ? 1d : 0d;

                // Check if the binary representation matches the target
                if (binary != targetMatrix[0][index])
                {
                    // Update target index if the predicted value needs to move closer to the optimal threshold
                    if (binary == 0d && Math.Abs(item - threshold) > maxDiff)
                    {
                        this.targetIndex = index;
                        maxDiff = Math.Abs(item - threshold);
                    }
                }

                // For early stopping edge case: find the '1' closest to the optimal threshold
                if (binary == 1d)
                {
                    double distance = Math.Abs(predicted[0][index] - optimalThreshold);
                    if (distance < closestOne)
                    {
                        closestOne = distance;
                        closestIndex = index;
                    }
                }
            }

            // If early stopping conditions are not met, set targetIndex to the closest '1'
            if (this.targetIndex == -1 && gap < differences.Average())
            {
                this.targetIndex = closestIndex;
            }

            // Compute the final loss based on variance
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
                // implements early stopping if the target index is -1
                dOutput[0][i] = (Lambda * ((4.0 / this.predicted.Cols) *
                    (this.variance - this.trueVariance) *
                    (this.predicted[0][i] - this.mean))) + (Alpha * (this.targetIndex == i ? -1 / (this.predicted[0][i] + Epsilon) : 0d));
            }

            return dOutput;
        }
    }
}
