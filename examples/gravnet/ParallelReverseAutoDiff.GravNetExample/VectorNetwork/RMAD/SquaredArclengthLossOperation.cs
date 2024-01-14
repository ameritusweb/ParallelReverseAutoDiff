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
        public Matrix Forward(Matrix predictions, double targetAngle)
        {
            var xOutput = predictions[0, 0];
            var yOutput = predictions[0, 1];

            double magnitude = Math.Sqrt(xOutput * xOutput + yOutput * yOutput);

            var xTarget = Math.Cos(targetAngle) * magnitude;
            this.xTarget = xTarget;
            var yTarget = Math.Sin(targetAngle) * magnitude;
            this.yTarget = yTarget;

            var radius = magnitude;
            this.radius = radius;

            // Calculate the dot product of the output and target vectors
            double dotProduct = xOutput * xTarget + yOutput * yTarget;
            this.dotProduct = dotProduct;

            // Normalize the dot product by the square of the radius
            double normalizedDotProduct = dotProduct / (radius * radius);

            // Clamp the normalized dot product to the range [-1, 1] to avoid numerical issues with arccos
            normalizedDotProduct = Math.Clamp(normalizedDotProduct, -1.0, 1.0);

            // Compute the angular difference using arccosine of the normalized dot product
            double theta = Math.Acos(normalizedDotProduct);

            // Compute the squared magnitude of the loss
            double lossMagnitude = Math.Pow(radius * theta, 2);

            var output = new Matrix(1, 1);
            output[0, 0] = lossMagnitude;

            return output;
        }

        /// <summary>
        /// Runs the backward operation for the mean squared error loss function.
        /// </summary>
        /// <returns>The gradient with respect to the predictions.</returns>
        public Matrix Backward()
        {
            Matrix dPredictions = new Matrix(1, 2);
            dPredictions[0, 0] = GradientWrtXOutput();
            dPredictions[0, 1] = GradientWrtYOutput();
            return dPredictions;
        }

        public double GradientWrtXOutput()
        {
            double normalizedDotProduct = dotProduct / (radius * radius);
            normalizedDotProduct = Math.Clamp(normalizedDotProduct, -1.0, 1.0);

            double theta = Math.Acos(normalizedDotProduct);
            double denominator = Math.Sqrt(1 - normalizedDotProduct * normalizedDotProduct);

            double gradXOutput = -2 * xTarget * theta / denominator;
            return gradXOutput;
        }

        public double GradientWrtYOutput()
        {
            double normalizedDotProduct = dotProduct / (radius * radius);
            normalizedDotProduct = Math.Clamp(normalizedDotProduct, -1.0, 1.0);

            double theta = Math.Acos(normalizedDotProduct);
            double denominator = Math.Sqrt(1 - normalizedDotProduct * normalizedDotProduct);

            double gradYOutput = -2 * yTarget * theta / denominator;
            return gradYOutput;
        }
    }
}
