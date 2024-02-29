//------------------------------------------------------------------------------
// <copyright file="SquaredArclengthEuclideanLossOperation.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Squared arclength euclidean loss operation.
    /// </summary>
    public class SquaredArclengthEuclideanLossOperation
    {
        private double dotProduct;
        private double xOutput;
        private double yOutput;
        private double xTarget;
        private double yTarget;
        private double radius;
        private double maxMagnitude;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static SquaredArclengthEuclideanLossOperation Instantiate(NeuralNetwork net)
        {
            return new SquaredArclengthEuclideanLossOperation();
        }

        /// <summary>
        /// Performs the forward operation for the squared arc length Euclidean distance loss function.
        /// </summary>
        /// <param name="predictions">The predictions matrix.</param>
        /// <param name="targetAngle">The target angle.</param>
        /// <returns>The scalar loss value.</returns>
        public Matrix Forward(Matrix predictions, double targetAngle)
        {
            var xOutput = predictions[0, 0];
            this.xOutput = xOutput;
            var yOutput = predictions[0, 1];
            this.yOutput = yOutput;

            // Maximum magnitude achievable by the summation of 225 unit vectors
            double maxMagnitude = 225;

            // Adjust target X and Y to represent the maximum magnitude in the target direction
            var xTarget = Math.Cos(targetAngle) * maxMagnitude;
            this.xTarget = xTarget;
            var yTarget = Math.Sin(targetAngle) * maxMagnitude;
            this.yTarget = yTarget;

            // Recalculate the radius (magnitude) of the output vector
            var radius = Math.Sqrt(xOutput * xOutput + yOutput * yOutput);
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

            double distanceXQuad = (0.75d * Math.Pow(xOutput, 2)) - (1.5d * xOutput * xTarget);
            
            double distanceYQuad = (0.75d * Math.Pow(yOutput, 2)) - (1.5d * yOutput * yTarget);
            double distanceAccum = distanceXQuad + distanceYQuad;

            // Example addition to the Forward method to emphasize magnitude
            double actualMagnitude = Math.Sqrt(Math.Pow(xOutput, 2) + Math.Pow(yOutput, 2));
            double magnitudeDiscrepancy = Math.Pow(maxMagnitude - actualMagnitude, 2);

            // Compute the squared magnitude of the loss
            double lossMagnitude = (Math.Pow(radius * theta, 2) + distanceAccum + magnitudeDiscrepancy) / 3d;

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

            // Calculate the additional magnitude discrepancy gradients
            double actualMagnitude = Math.Sqrt((this.xOutput * this.xOutput) + (this.yOutput * this.yOutput));
            double magDiscrepancyGradient = -2 * (maxMagnitude - actualMagnitude);

            double dMagDiscrepancy_dX = magDiscrepancyGradient * (xOutput / actualMagnitude);
            double dMagDiscrepancy_dY = magDiscrepancyGradient * (yOutput / actualMagnitude);

            dPredictions[0, 0] = (-1d * gradX) + eX + dMagDiscrepancy_dX;
            dPredictions[0, 1] = gradY + eY + dMagDiscrepancy_dY;
            return dPredictions;
        }

        public (double GradX, double GradY) EuclideanGradientWrtOutput()
        {
            double X = this.xOutput;
            double Y = this.yOutput;

            double dLoss_dX = (X - this.xTarget) * (3d/2d);
            double dLoss_dY = (Y - this.yTarget) * (3d/2d);
            return (dLoss_dX, dLoss_dY);
        }

        public double GradientWrtXOutput()
        {
            double normalizedDotProduct = this.dotProduct / (this.radius * this.radius);
            normalizedDotProduct = Math.Clamp(normalizedDotProduct, -1.0, 1.0);

            double theta = Math.Acos(normalizedDotProduct);
            double denominator = Math.Sqrt(1 - (normalizedDotProduct * normalizedDotProduct));

            double gradXOutput = this.xTarget * theta / denominator;
            return gradXOutput;
        }

        public double GradientWrtYOutput()
        {
            double normalizedDotProduct = this.dotProduct / (this.radius * this.radius);
            normalizedDotProduct = Math.Clamp(normalizedDotProduct, -1.0, 1.0);

            double theta = Math.Acos(normalizedDotProduct);
            double denominator = Math.Sqrt(1 - (normalizedDotProduct * normalizedDotProduct));

            double gradYOutput = this.yTarget * theta / denominator;
            return gradYOutput;
        }
    }
}
