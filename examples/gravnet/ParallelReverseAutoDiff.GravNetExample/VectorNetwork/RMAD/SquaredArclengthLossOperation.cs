﻿//------------------------------------------------------------------------------
// <copyright file="SquaredArclengthLossOperation.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Squared arclength loss operation.
    /// </summary>
    public class SquaredArclengthLossOperation
    {
        private double gradientX;
        private double gradientY;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static SquaredArclengthLossOperation Instantiate(NeuralNetwork net)
        {
            return new SquaredArclengthLossOperation();
        }

        /// <summary>
        /// Performs the forward operation for the mean squared error loss function.
        /// </summary>
        /// <param name="predictions">The predictions matrix.</param>
        /// <param name="targetAngle">The target angle.</param>
        /// <returns>The scalar loss value.</returns>
        public Matrix Forward(Matrix predictions, double targetAngle, double radius = 10d)
        {
            var xOutput = predictions[0, 0];
            var yOutput = predictions[0, 1];

            var outputTheta = Math.Atan2(yOutput, xOutput);
            var xOutputMapped = radius * Math.Cos(outputTheta);
            var yOutputMapped = radius * Math.Sin(outputTheta);

            var xTarget = radius * Math.Cos(targetAngle);

            var yTarget = radius * Math.Sin(targetAngle);

            // Calculate the dot product of the output and target vectors
            double dotProduct = (xOutputMapped * xTarget) + (yOutputMapped * yTarget);

            // Normalize the dot product by the square of the radius
            double normalizedDotProduct = dotProduct / (radius * radius);

            // Clamp the normalized dot product to the range [-1, 1] to avoid numerical issues with arccos
            normalizedDotProduct = Math.Clamp(normalizedDotProduct, -1.0, 1.0);

            // Compute the angular difference using arccosine of the normalized dot product
            double theta = Math.Acos(normalizedDotProduct);

            double xDiffSign = Math.Sign(xTarget - xOutputMapped);

            double lossX = radius * theta * xDiffSign;

            double yDiffSign = Math.Sign(yTarget - yOutputMapped);

            double lossY = radius * theta * yDiffSign;

            double lossXCubedDivide = lossX * lossX * lossX / 3d;

            double lossYCubedDivide = lossY * lossY * lossY / 3d;

            // Gradient calculations //
            double dXOutputMapped_dX = -radius * Math.Sin(outputTheta) * (-yOutput / ((xOutput * xOutput) + (yOutput * yOutput)));

            double dYOutputMapped_dY = radius * Math.Cos(outputTheta) * (1 / ((xOutput * xOutput) + (yOutput * yOutput)));

            double dNormalizedDotProduct_dXOutputMapped = xTarget / (radius * radius);

            double dNormalizedDotProduct_dYOutputMapped = yTarget / (radius * radius);

            double dLossX_dNormalizedDotProduct = radius * -1 / Math.Sqrt(1 - (normalizedDotProduct * normalizedDotProduct));

            double dLossY_dNormalizedDotProduct = radius * -1 / Math.Sqrt(1 - (normalizedDotProduct * normalizedDotProduct));

            double dLossXCubedDivide_dLossX = lossX * lossX;

            double dLossYCubedDivide_dLossY = lossY * lossY;

            this.gradientX = (targetAngle < (Math.PI / 2) ? -1d : 1d) * (yOutputMapped < 0.0d ? 1d : -1d) * (lossXCubedDivide < 0.0d ? -1d : 1d) * dLossXCubedDivide_dLossX * dLossX_dNormalizedDotProduct * dNormalizedDotProduct_dXOutputMapped * dXOutputMapped_dX;
            this.gradientY = (targetAngle < 0 ? -1d : 1d) * (xOutputMapped < 0.0d ? -1d : 1d) * (lossYCubedDivide < 0.0d ? -1d : 1d) * dLossYCubedDivide_dLossY * dLossY_dNormalizedDotProduct * dNormalizedDotProduct_dYOutputMapped * dYOutputMapped_dY;
            ////////////////////////////////

            var output = new Matrix(1, 2);
            output[0, 0] = lossXCubedDivide;
            output[0, 1] = lossYCubedDivide;

            return output;
        }

        /// <summary>
        /// Runs the backward operation for the mean squared error loss function.
        /// </summary>
        /// <returns>The gradient with respect to the predictions.</returns>
        public Matrix Backward()
        {
            Matrix dPredictions = new Matrix(1, 2);
            dPredictions[0, 0] = this.gradientX;
            dPredictions[0, 1] = this.gradientY;
            return dPredictions;
        }
    }
}
