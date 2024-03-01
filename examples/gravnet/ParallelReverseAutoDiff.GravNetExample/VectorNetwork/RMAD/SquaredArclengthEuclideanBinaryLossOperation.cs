//------------------------------------------------------------------------------
// <copyright file="SquaredArclengthEuclideanBinaryLossOperation.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Squared arclength euclidean loss operation.
    /// </summary>
    public class SquaredArclengthEuclideanBinaryLossOperation
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
        public static SquaredArclengthEuclideanBinaryLossOperation Instantiate(NeuralNetwork net)
        {
            return new SquaredArclengthEuclideanBinaryLossOperation();
        }

        /// <summary>
        /// Performs the forward operation for the squared arc length Euclidean distance loss function.
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

            double distanceX = Math.Pow(xOutput - xTarget, 3);

            double distanceY = Math.Pow(yOutput - yTarget, 3);
            double distanceCubed = distanceX + distanceY;

            // Compute the squared magnitude of the loss
            double lossMagnitude = (Math.Pow(radius * theta, 2) + distanceCubed) / 2d;

            var output = new Matrix(1, 1);
            output[0, 0] = lossMagnitude;

            return output;
        }

        /// <summary>
        /// Runs the backward operation for the squared arclength Euclidean distance loss function.
        /// </summary>
        /// <returns>The gradient with respect to the predictions.</returns>
        public Matrix Backward()
        {
            Matrix dPredictions = new Matrix(1, 2);
            var gradX = GradientWrtXOutput();
            var gradY = GradientWrtYOutput();
            var (eX, eY) = EuclideanGradientWrtOutput();
            dPredictions[0, 0] = (-1d * gradX) + eX;
            dPredictions[0, 1] = gradY + eY;
            return dPredictions;
        }

        public (double GradX, double GradY) EuclideanGradientWrtOutput()
        {
            double X = this.xOutput;
            double Y = this.yOutput;

            double dLoss_dX = (X - xTarget) * (3d / 2d);
            double dLoss_dY = (Y - yTarget) * (3d / 2d);
            return (dLoss_dX, dLoss_dY);
        }

        public double GradientWrtXOutput()
        {
            double normalizedDotProduct = dotProduct / (radius * radius);
            normalizedDotProduct = Math.Clamp(normalizedDotProduct, -1.0, 1.0);

            double theta = Math.Acos(normalizedDotProduct);
            double denominator = Math.Sqrt(1 - normalizedDotProduct * normalizedDotProduct);

            double gradXOutput = xTarget * theta / denominator;
            return gradXOutput;
        }

        public double GradientWrtYOutput()
        {
            double normalizedDotProduct = dotProduct / (radius * radius);
            normalizedDotProduct = Math.Clamp(normalizedDotProduct, -1.0, 1.0);

            double theta = Math.Acos(normalizedDotProduct);
            double denominator = Math.Sqrt(1 - normalizedDotProduct * normalizedDotProduct);

            double gradYOutput = yTarget * theta / denominator;
            return gradYOutput;
        }
    }
}
