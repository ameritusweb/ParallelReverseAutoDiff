//------------------------------------------------------------------------------
// <copyright file="VarianceLossOperation.cs" author="ameritusweb" date="9/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FsmnnExample.FiniteStateMachine.RMAD
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Variance loss operation.
    /// </summary>
    public class VarianceLossOperation
    {
        private Matrix predicted;
        private double trueVariance;
        private double mean;
        private double variance;

        /// <summary>
        /// Performs the forward operation.
        /// </summary>
        /// <param name="predicted">1xN matrix of predicted probabilities.</param>
        /// <param name="trueVariance">1xN matrix of true labels (one-hot encoded).</param>
        /// <returns>The computed loss.</returns>
        public double Forward(Matrix predicted, double trueVariance)
        {
            this.predicted = predicted;
            this.trueVariance = trueVariance;

            double mean = predicted[0].Average();
            this.mean = mean;

            double variance = predicted[0].Select(x => (x - mean) * (x - mean)).Average();
            this.variance = variance;

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
                dOutput[0][i] = 0.0001d / ((4.0 / this.predicted.Cols) * (this.variance - this.trueVariance) * (this.predicted[0][i] - this.mean));
            }

            return dOutput;
        }
    }
}
