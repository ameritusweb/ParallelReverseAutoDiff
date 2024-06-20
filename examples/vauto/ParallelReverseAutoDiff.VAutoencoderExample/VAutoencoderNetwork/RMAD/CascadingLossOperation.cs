//------------------------------------------------------------------------------
// <copyright file="CascadingLossOperation.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.VGruExample.VGruNetwork.RMAD
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Cascading loss operation.
    /// </summary>
    public class CascadingLossOperation
    {
        private Matrix predictions;
        private double targetAngle;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static CascadingLossOperation Instantiate(NeuralNetwork net)
        {
            return new CascadingLossOperation();
        }

        /// <summary>
        /// Performs the forward operation for the cascading loss function.
        /// </summary>
        /// <param name="predictions">The predictions matrix.</param>
        /// <param name="targetAngle">The target angle.</param>
        /// <returns>The scalar loss value.</returns>
        public Matrix Forward(Matrix predictions, double targetAngle)
        {
            this.predictions = predictions;
            this.targetAngle = targetAngle;

            var cascadingFactor = this.CalculateCascadingFactor(targetAngle);

            var loss = 0d;

            for (int i = 0; i < predictions.Rows - 1; ++i)
            {
                for (int j = 0; j < predictions.Cols / 2; ++j)
                {
                    var expectedValue = predictions[i, j] * cascadingFactor;
                    var actualValue = predictions[i + 1, j];
                    loss += Math.Pow(expectedValue - actualValue, 2);
                }
            }

            var averageLoss = loss / ((predictions.Rows - 1) * predictions.Cols / 2);

            var output = new Matrix(1, 1);
            output[0, 0] = averageLoss;

            return output;
        }

        /// <summary>
        /// Runs the backward operation for the cascading loss function.
        /// </summary>
        /// <returns>The gradient with respect to the predictions.</returns>
        public Matrix Backward()
        {
            Matrix dPredictions = new Matrix(this.predictions.Rows, this.predictions.Cols);

            var cascadingFactor = this.CalculateCascadingFactor(this.targetAngle);

            for (int i = 0; i < this.predictions.Rows - 1; ++i)
            {
                for (int j = 0; j < this.predictions.Cols / 2; ++j)
                {
                    double predictedValue = this.predictions[i, j];
                    double nextRowValue = this.predictions[i + 1, j];
                    double expectedValue = predictedValue * cascadingFactor;

                    // Gradient for the current prediction based on its influence on the expected next row value
                    dPredictions[i, j] += 2 * (expectedValue - nextRowValue) * cascadingFactor;

                    // Gradient for the next row prediction based on the discrepancy with the current expected value
                    dPredictions[i + 1, j] -= 2 * (expectedValue - nextRowValue);
                }
            }

            // Normalize the gradients by the number of elements contributing to each
            for (int i = 0; i < this.predictions.Rows; ++i)
            {
                for (int j = 0; j < this.predictions.Cols / 2; ++j)
                {
                    dPredictions[i, j] /= (this.predictions.Cols / 2) * (this.predictions.Rows - 1);
                }
            }

            return dPredictions;
        }

        private double CalculateCascadingFactor(double targetAngle)
        {
            return 1 + ((targetAngle + (Math.PI / 2d)) / Math.PI);
        }
    }
}
