//------------------------------------------------------------------------------
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
        private double dotProduct;
        private double xTarget;
        private double yTarget;
        private double radius;
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
            predictions[0, 1] += 0.001d;
            var xOutput = predictions[0, 0];
            var yOutput = predictions[0, 1];

            var outputTheta = Math.Atan2(yOutput, xOutput);
            var xOutputMapped = radius * Math.Cos(outputTheta);
            var yOutputMapped = radius * Math.Sin(outputTheta);

            var xTarget = radius * Math.Cos(targetAngle);
            this.xTarget = xTarget;
            var yTarget = radius * Math.Sin(targetAngle);
            this.yTarget = yTarget;

            this.radius = radius;

            // Calculate the dot product of the output and target vectors
            double dotProduct = (xOutputMapped * xTarget) + (yOutputMapped * yTarget);
            this.dotProduct = dotProduct;

            // Normalize the dot product by the square of the radius
            double normalizedDotProduct = dotProduct / (radius * radius);

            // Clamp the normalized dot product to the range [-1, 1] to avoid numerical issues with arccos
            normalizedDotProduct = Math.Clamp(normalizedDotProduct, -1.0, 1.0);

            // Compute the angular difference using arccosine of the normalized dot product
            double theta = Math.Acos(normalizedDotProduct);

            double xDiff = Math.Sign(xOutputMapped - xTarget);

            double lossX = radius * theta * xDiff;

            double yDiff = Math.Sign(yOutputMapped - yTarget);

            double lossY = radius * theta * yDiff;

            double lossXCubed = lossX * lossX * lossX;

            double lossYCubed = lossY * lossY * lossY;

            // Gradient calculations //
            double dXOutputMapped_dX = -radius * Math.Sin(outputTheta) * (-yOutput / ((xOutput * xOutput) + (yOutput * yOutput)));

            double dXOutputMapped_dY = radius * Math.Cos(outputTheta) * (1 / ((xOutput * xOutput) + (yOutput * yOutput)));

            double dNormalizedDotProduct_dXOutputMapped = xTarget / (radius * radius);

            double dNormalizedDotProduct_dYOutputMapped = yTarget / (radius * radius);

            double dLossX_dNormalizedDotProduct = radius * xDiff * -1 / Math.Sqrt(1 - (normalizedDotProduct * normalizedDotProduct));

            double dLossY_dNormalizedDotProduct = radius * yDiff * -1 / Math.Sqrt(1 - (normalizedDotProduct * normalizedDotProduct));

            this.gradientX = (lossXCubed < 0.0d ? -1d : 1d) * dLossX_dNormalizedDotProduct * dNormalizedDotProduct_dXOutputMapped * dXOutputMapped_dX;
            this.gradientY = (lossYCubed < 0.0d ? -1d : 1d) * dLossY_dNormalizedDotProduct * dNormalizedDotProduct_dYOutputMapped * dXOutputMapped_dY;
            ////////////////////////////////

            var output = new Matrix(1, 2);
            output[0, 0] = lossXCubed;
            output[0, 1] = lossYCubed;

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
