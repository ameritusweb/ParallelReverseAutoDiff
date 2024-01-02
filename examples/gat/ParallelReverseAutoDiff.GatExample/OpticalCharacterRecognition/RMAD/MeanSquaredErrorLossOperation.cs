//------------------------------------------------------------------------------
// <copyright file="MeanSquaredErrorLossOperation.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GatExample.OpticalCharacterRecognition.RMAD
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Mean squared error loss operation.
    /// </summary>
    public class MeanSquaredErrorLossOperation
    {
        private Matrix predictions;
        private Matrix targets;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static MeanSquaredErrorLossOperation Instantiate(NeuralNetwork net)
        {
            return new MeanSquaredErrorLossOperation();
        }

        /// <summary>
        /// Performs the forward operation for the mean squared error loss function.
        /// </summary>
        /// <param name="predictions">The predictions matrix.</param>
        /// <param name="targets">The target values matrix.</param>
        /// <returns>The scalar loss value.</returns>
        public Matrix Forward(Matrix predictions, Matrix targets)
        {
            this.predictions = predictions;
            this.targets = targets;

            // Validate dimensions
            if (predictions.Rows != targets.Rows || predictions.Cols != targets.Cols)
            {
                throw new InvalidOperationException("Predictions and targets must have the same dimensions.");
            }

            Matrix diff = predictions - targets;
            Matrix squaredDiff = diff.ElementwisePower(2);
            double mse = squaredDiff.Mean();

            var output = new Matrix(1, 1);
            output[0, 0] = mse;

            return output;
        }

        /// <summary>
        /// Runs the backward operation for the mean squared error loss function.
        /// </summary>
        /// <returns>The gradient with respect to the predictions.</returns>
        public Matrix Backward()
        {
            // Gradient of MSE wrt predictions
            int n = this.predictions.Rows * this.predictions.Cols;
            Matrix dPredictions = (this.predictions - this.targets) * (2.0 / n);

            // Implementing dropout
            Matrix dropoutMask = GenerateDropoutMask(dPredictions.Rows, dPredictions.Cols, 0.2);
            Matrix dPredictionsWithDropout = dPredictions.ElementwiseMultiply(dropoutMask);

            return dPredictionsWithDropout;
        }

        /// <summary>
        /// Generates a dropout mask with a specified dropout rate.
        /// </summary>
        /// <param name="rows">Number of rows in the mask.</param>
        /// <param name="cols">Number of columns in the mask.</param>
        /// <param name="dropoutRate">Dropout rate (fraction of elements to be zeroed).</param>
        /// <returns>Dropout mask matrix.</returns>
        private Matrix GenerateDropoutMask(int rows, int cols, double dropoutRate)
        {
            Random rand = new Random();
            Matrix mask = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    mask[i, j] = rand.NextDouble() > dropoutRate ? 1 : 0;
                }
            }

            return mask;
        }
    }
}
