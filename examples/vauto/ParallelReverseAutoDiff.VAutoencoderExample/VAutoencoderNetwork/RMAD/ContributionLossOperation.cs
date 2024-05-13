//------------------------------------------------------------------------------
// <copyright file="ContributionLossOperation.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.VGruExample.VGruNetwork.RMAD
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Contribution loss operation.
    /// </summary>
    public class ContributionLossOperation
    {
        private Matrix predictions;
        private double targetAngle;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static ContributionLossOperation Instantiate(NeuralNetwork net)
        {
            return new ContributionLossOperation();
        }

        /// <summary>
        /// Performs the forward operation for the contribution loss function.
        /// </summary>
        /// <param name="predictions">The predictions matrix.</param>
        /// <param name="targetAngle">The target angle.</param>
        /// <returns>The scalar loss value.</returns>
        public Matrix Forward(Matrix predictions, double targetAngle)
        {
            this.predictions = predictions;
            this.targetAngle = targetAngle;

            var matrix = VectorToMatrix.CreateLine(targetAngle, 11);

            var loss = 0d;

            for (int i = 0; i < predictions.Rows; ++i)
            {
                for (int j = 0; j < predictions.Cols / 2d; ++j)
                {
                    if (matrix[i, j] > 0.5)
                    {
                        loss -= predictions[i, j] * predictions[i, j];
                    }
                    else
                    {
                        loss += predictions[i, j] * predictions[i, j];
                    }
                }
            }

            var output = new Matrix(1, 1);
            output[0, 0] = loss;

            return output;
        }

        /// <summary>
        /// Runs the backward operation for the contribution loss function.
        /// </summary>
        /// <returns>The gradient with respect to the predictions.</returns>
        public Matrix Backward()
        {
            Matrix dPredictions = new Matrix(this.predictions.Rows, this.predictions.Cols);
            var matrix = VectorToMatrix.CreateLine(this.targetAngle, 11); // Recreating the line matrix for backpropagation

            for (int i = 0; i < this.predictions.Rows; ++i)
            {
                for (int j = 0; j < this.predictions.Cols / 2d; ++j)
                {
                    // Compute the gradient based on the squared condition in the forward pass
                    if (matrix[i, j] > 0.5)
                    {
                        // If the condition is true, we subtracted the square of the prediction from the loss
                        // The derivative of -x^2 is -2x, hence the negative sign
                        dPredictions[i, j] = -2 * this.predictions[i, j];
                    }
                    else
                    {
                        // If the condition is false, we added the square of the prediction to the loss
                        // The derivative of x^2 is 2x
                        dPredictions[i, j] = 2 * this.predictions[i, j];
                    }
                }
            }

            return dPredictions;
        }
    }
}
