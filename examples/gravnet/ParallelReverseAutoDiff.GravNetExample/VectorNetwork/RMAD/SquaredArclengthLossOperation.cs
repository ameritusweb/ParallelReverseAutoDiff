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
        private double xOutput;
        private double yOutput;
        private double xTarget;
        private double yTarget;
        private double radius;
        private double normalizedDotProduct;
        private double theta;
        private double magnitude;
        private double targetAngle;

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
            this.targetAngle = targetAngle;

            var xOutput = predictions[0, 0];
            this.xOutput = xOutput;
            var yOutput = predictions[0, 1];
            this.yOutput = yOutput;

            double magnitude = Math.Sqrt(xOutput * xOutput + yOutput * yOutput);
            this.magnitude = magnitude;

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
            this.normalizedDotProduct = normalizedDotProduct;

            // Compute the angular difference using arccosine of the normalized dot product
            double theta = Math.Acos(normalizedDotProduct);
            this.theta = theta;

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
            var gradX = GradientWrtXOutput();
            var gradY = GradientWrtYOutput();
            (dPredictions[0, 0], dPredictions[0, 1]) = GradientWrtOutput();
            dPredictions[0, 0] = gradX;
            dPredictions[0, 1] = gradY;
            return dPredictions;
        }

        public (double GradX, double GradY) GradientWrtOutput()
        {
            double X = this.xOutput;
            double Y = this.yOutput;

            // double dMagnitude_dX = X / this.magnitude;
            // double dXTarget_dMagnitude = Math.Cos(this.targetAngle);

            // double dMagnitude_dY = Y / this.magnitude;
            // double dYTarget_dMagnitude = Math.Sin(this.targetAngle);

            double dDotProduct_dX = (X * (X * Math.Cos(this.targetAngle) / this.magnitude)) + this.xTarget; // dXOutputTimesXTarget_dXOutput
            double dDotProduct_dY = (Y * (Y * Math.Sin(this.targetAngle) / this.magnitude)) + this.yTarget; // dYOutputTimesYTarget_dYOutput

            // double dNormalizedDotProduct_dDotProduct = 1d / (this.radius * this.radius);
            double dTheta_dNormalizedDotProduct = -1d / Math.Sqrt(1d - (this.normalizedDotProduct * this.normalizedDotProduct));
            // double dLossMagnitude_dTheta = 2d * (this.radius * this.radius) * this.theta;

            // Simplified:
            double dLossMagnitude_dX = -2d * this.theta * dTheta_dNormalizedDotProduct * dDotProduct_dX;
            double dLossMagnitude_dY = -2d * this.theta * dTheta_dNormalizedDotProduct * dDotProduct_dY;
            return (dLossMagnitude_dX, dLossMagnitude_dY);
        }

        public double GradientWrtXOutput()
        {
            double normalizedDotProduct = dotProduct / (radius * radius);
            normalizedDotProduct = Math.Clamp(normalizedDotProduct, -1.0, 1.0);

            double theta = Math.Acos(normalizedDotProduct);
            double denominator = Math.Sqrt(1 - normalizedDotProduct * normalizedDotProduct);

            double gradXOutput = 2 * xTarget * theta / denominator;
            return gradXOutput;
        }

        public double GradientWrtYOutput()
        {
            double normalizedDotProduct = dotProduct / (radius * radius);
            normalizedDotProduct = Math.Clamp(normalizedDotProduct, -1.0, 1.0);

            double theta = Math.Acos(normalizedDotProduct);
            double denominator = Math.Sqrt(1 - normalizedDotProduct * normalizedDotProduct);

            double gradYOutput = 2 * yTarget * theta / denominator;
            return gradYOutput;
        }
    }
}
