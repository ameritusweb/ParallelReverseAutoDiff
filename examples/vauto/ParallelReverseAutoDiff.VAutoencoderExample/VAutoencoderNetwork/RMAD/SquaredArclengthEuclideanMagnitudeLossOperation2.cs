//------------------------------------------------------------------------------
// <copyright file="SquaredArclengthEuclideanMagnitudeLossOperation2.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.VGruExample.VGruNetwork.RMAD
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Squared arclength euclidean magnitude loss operation.
    /// </summary>
    public class SquaredArclengthEuclideanMagnitudeLossOperation2
    {
        private double dotProduct;
        private double xOutput;
        private double yOutput;
        private double xTarget;
        private double yTarget;
        private double xTargetUnnormalized;
        private double yTargetUnnormalized;
        private double radius;
        private double actualAngle;
        private double targetAngle;
        private double targetMagnitude;
        private CalculatedValues calculatedValues;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static SquaredArclengthEuclideanMagnitudeLossOperation2 Instantiate(NeuralNetwork net)
        {
            return new SquaredArclengthEuclideanMagnitudeLossOperation2();
        }

        /// <summary>
        /// Performs the forward operation for the squared arc length Euclidean distance loss function.
        /// </summary>
        /// <param name="predictions">The predictions matrix.</param>
        /// <param name="targetAngle">The target angle.</param>
        /// <param name="targetMagnitude">The target magnitude.</param>
        /// <returns>The scalar loss value.</returns>
        public Matrix Forward(Matrix predictions, double targetAngle, double targetMagnitude)
        {
            this.targetAngle = targetAngle;
            var xOutput = predictions[0, 0];
            this.xOutput = xOutput;
            var yOutput = predictions[0, 1];
            this.yOutput = yOutput;

            this.targetMagnitude = targetMagnitude;

            double magnitude = Math.Sqrt((xOutput * xOutput) + (yOutput * yOutput));
            this.actualAngle = Math.Atan2(yOutput, xOutput);

            var xTarget = Math.Cos(targetAngle) * magnitude;
            this.xTarget = xTarget;
            var yTarget = Math.Sin(targetAngle) * magnitude;
            this.yTarget = yTarget;

            this.xTargetUnnormalized = Math.Cos(targetAngle) * targetMagnitude;
            this.yTargetUnnormalized = Math.Sin(targetAngle) * targetMagnitude;

            var xOutputNormalized = Math.Cos(this.actualAngle) * targetMagnitude;
            var yOutputNormalized = Math.Sin(this.actualAngle) * targetMagnitude;

            var dXOutputNormalized_dXOutput = -Math.Sin(this.actualAngle) * (-yOutput / magnitude) * targetMagnitude;
            var dXOutputNormalized_dYOutput = -Math.Sin(this.actualAngle) * (xOutput / magnitude) * targetMagnitude;
            var dYOutputNormalized_dXOutput = Math.Cos(this.actualAngle) * (-yOutput / magnitude) * targetMagnitude;
            var dYOutputNormalized_dYOutput = Math.Cos(this.actualAngle) * (xOutput / magnitude) * targetMagnitude;

            var radius = targetMagnitude;
            this.radius = radius;

            // Calculate the dot product of the output and target vectors
            double dotProduct = (xOutputNormalized * this.xTargetUnnormalized) + (yOutputNormalized * this.yTargetUnnormalized);
            this.dotProduct = dotProduct;

            var dDotProduct_dXOutputNormalized = xTarget;
            var dDotProduct_dYOutputNormalized = yTarget;

            // Normalize the dot product by the square of the radius
            double normalizedDotProduct = dotProduct / (radius * radius);

            var dNormalizedDotProduct_dDotProduct = 1 / (radius * radius);

            var dNormalizedDotProduct_dXOutput = dNormalizedDotProduct_dDotProduct *
                ((dDotProduct_dXOutputNormalized * dXOutputNormalized_dXOutput)
                +
                (dDotProduct_dYOutputNormalized * dYOutputNormalized_dXOutput));

            var dNormalizedDotProduct_dYOutput = dNormalizedDotProduct_dDotProduct *
                ((dDotProduct_dXOutputNormalized * dXOutputNormalized_dYOutput)
                +
                (dDotProduct_dYOutputNormalized * dYOutputNormalized_dYOutput));

            // Clamp the normalized dot product to the range [-1, 1] to avoid numerical issues with arccos
            normalizedDotProduct = Math.Clamp(normalizedDotProduct, -1.0, 1.0);

            // Compute the angular difference using arccosine of the normalized dot product
            double theta = Math.Acos(normalizedDotProduct);

            var dTheta_dXOutput = dNormalizedDotProduct_dXOutput * -1 / Math.Sqrt(1 - (normalizedDotProduct * normalizedDotProduct));
            var dTheta_dYOutput = dNormalizedDotProduct_dYOutput * -1 / Math.Sqrt(1 - (normalizedDotProduct * normalizedDotProduct));

            double distanceXQuad = (0.75d * Math.Pow(xOutput, 2)) - (1.5d * xOutput * this.xTargetUnnormalized);

            double distanceYQuad = (0.75d * Math.Pow(yOutput, 2)) - (1.5d * yOutput * this.yTargetUnnormalized);
            double distanceAccum = distanceXQuad + distanceYQuad;

            // Example addition to the Forward method to emphasize magnitude
            double actualMagnitude = Math.Sqrt(Math.Pow(xOutput, 2) + Math.Pow(yOutput, 2));
            double magnitudeDiscrepancy = Math.Pow(targetMagnitude - actualMagnitude, 2);

            double arcLength = Math.Pow(radius * theta, 2);

            var dArcLength_dXOutput = 2 * radius * theta * radius * dTheta_dXOutput;
            var dArcLength_dYOutput = 2 * radius * theta * radius * dTheta_dYOutput;
            this.calculatedValues = new CalculatedValues
            {
                CV_dArcLength_dXOutput = dArcLength_dXOutput,
                CV_dArcLength_dYOutput = dArcLength_dYOutput,
            };

            // Compute the squared magnitude of the loss
            double lossMagnitude = (arcLength + distanceAccum + magnitudeDiscrepancy) / 3d;

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
            var gradX = this.GradientWrtXOutput();
            var gradY = this.GradientWrtYOutput();
            var (eX, eY) = this.EuclideanGradientWrtOutput();

            // Calculate the additional magnitude discrepancy gradients
            double actualMagnitude = Math.Sqrt((this.xOutput * this.xOutput) + (this.yOutput * this.yOutput));
            double magDiscrepancyGradient = -2 * (this.targetMagnitude - actualMagnitude);

            double dMagDiscrepancy_dX = magDiscrepancyGradient * (this.xOutput / actualMagnitude);
            double dMagDiscrepancy_dY = magDiscrepancyGradient * (this.yOutput / actualMagnitude);

            (double cX, double cY) = this.CalculateGradientDirection();
            dPredictions[0, 0] = (cX * Math.Abs(gradX)) + eX + dMagDiscrepancy_dX;
            dPredictions[0, 1] = (cY * Math.Abs(gradY)) + eY + dMagDiscrepancy_dY;

            return dPredictions;
        }

        /// <summary>
        /// Calculate the Euclidean gradient with respect to the output.
        /// </summary>
        /// <returns>The Euclidean gradient.</returns>
        public (double GradX, double GradY) EuclideanGradientWrtOutput()
        {
            double x = this.xOutput;
            double y = this.yOutput;

            double dLoss_dX = (x - this.xTargetUnnormalized) * (3d / 2d);
            double dLoss_dY = (y - this.yTargetUnnormalized) * (3d / 2d);
            return (dLoss_dX, dLoss_dY);
        }

        /// <summary>
        /// Calculate the gradient with respect to the x output.
        /// </summary>
        /// <returns>The gradient.</returns>
        public double GradientWrtXOutput()
        {
            double gradXOutput = this.calculatedValues.CV_dArcLength_dXOutput;
            return gradXOutput;
        }

        /// <summary>
        /// Calculate the gradient with respect to the y output.
        /// </summary>
        /// <returns>The gradient.</returns>
        public double GradientWrtYOutput()
        {
            double gradYOutput = this.calculatedValues.CV_dArcLength_dYOutput;
            return gradYOutput;
        }

        private (double X, double Y) CalculateGradientDirection()
        {
            var actualAngle = this.actualAngle;
            var targetAngle = this.targetAngle;

            var quadrant = this.GetQuadrant(actualAngle);
            var targetQuadrant = this.GetQuadrant(targetAngle);
            var oppositeAngle = this.CalculateOppositeAngle(targetAngle);
            if (targetQuadrant == 1)
            {
                if (quadrant == 1)
                {
                    if (actualAngle < targetAngle)
                    {
                        return (1, -1); // x increase, y decrease
                    }
                    else
                    {
                        return (-1, 1); // x decrease, y increase
                    }
                }
                else if (quadrant == 2)
                {
                    return (-1, -1); // x decrease, y decrease
                }
                else if (quadrant == 3)
                {
                    if (actualAngle < oppositeAngle)
                    {
                        return (1, -1); // x increase, y decrease
                    }
                    else
                    {
                        return (-1, 1); // x decrease, y increase
                    }
                }
                else
                {
                    return (-1, -1); // x decrease, y decrease
                }
            }
            else if (targetQuadrant == 2)
            {
                if (quadrant == 1)
                {
                    return (1, -1); // x increase, y decrease
                }
                else if (quadrant == 2)
                {
                    if (actualAngle < targetAngle)
                    {
                        return (1, 1); // x increase, y increase
                    }
                    else
                    {
                        return (-1, -1); // x decrease, y decrease
                    }
                }
                else if (quadrant == 3)
                {
                    return (1, -1); // x increase, y decrease
                }
                else
                {
                    if (actualAngle < oppositeAngle)
                    {
                        return (1, 1); // x increase, y increase
                    }
                    else
                    {
                        return (-1, -1); // x decrease, y decrease
                    }
                }
            }
            else if (targetQuadrant == 3)
            {
                if (quadrant == 1)
                {
                    if (actualAngle < oppositeAngle)
                    {
                        return (-1, 1); // x decrease, y increase
                    }
                    else
                    {
                        return (1, -1); // x increase, y decrease
                    }
                }
                else if (quadrant == 2)
                {
                    return (1, 1); // x increase, y increase
                }
                else if (quadrant == 3)
                {
                    if (actualAngle < targetAngle)
                    {
                        return (-1, 1); // x decrease, y increase
                    }
                    else
                    {
                        return (1, -1); // x increase, y decrease
                    }
                }
                else
                {
                    return (1, 1); // x increase, y increase
                }
            }
            else if (targetQuadrant == 4)
            {
                if (quadrant == 1)
                {
                    return (-1, 1); // x decrease, y increase
                }
                else if (quadrant == 2)
                {
                    if (actualAngle < oppositeAngle)
                    {
                        return (-1, -1); // x decrease, y decrease
                    }
                    else
                    {
                        return (1, 1); // x increase, y increase
                    }
                }
                else if (quadrant == 3)
                {
                    return (-1, 1); // x decrease, y increase
                }
                else
                {
                    if (actualAngle < targetAngle)
                    {
                        return (-1, -1); // x decrease, y decrease
                    }
                    else
                    {
                        return (1, 1); // x increase, y increase
                    }
                }
            }

            throw new InvalidOperationException("Unsupported target quadrant");
        }

        private int GetQuadrant(double angleInRadians)
        {
            // Normalize the angle to be within 0 to 2pi radians
            double normalizedAngle = angleInRadians % (2 * Math.PI);

            // Adjust if negative to ensure it falls within the 0 to 2pi range
            if (normalizedAngle < 0)
            {
                normalizedAngle += 2 * Math.PI;
            }

            // Determine the quadrant
            if (normalizedAngle >= 0 && normalizedAngle < Math.PI / 2)
            {
                return 1;
            }
            else if (normalizedAngle >= Math.PI / 2 && normalizedAngle < Math.PI)
            {
                return 2;
            }
            else if (normalizedAngle >= Math.PI && normalizedAngle < 3 * Math.PI / 2)
            {
                return 3;
            }
            else
            {
                return 4;
            }
        }

        private double CalculateOppositeAngle(double targetAngle)
        {
            // Add π to the target angle to find the opposite angle
            double oppositeAngle = targetAngle + Math.PI;

            // Normalize the opposite angle to be within [-2π, 2π]
            if (oppositeAngle > 2 * Math.PI)
            {
                oppositeAngle -= 2 * Math.PI;
            }
            else if (oppositeAngle < -2 * Math.PI)
            {
                oppositeAngle += 2 * Math.PI;
            }

            return oppositeAngle;
        }

        private struct CalculatedValues
        {
            public double CV_dArcLength_dXOutput { get; internal set; }

            public double CV_dArcLength_dYOutput { get; internal set; }
        }
    }
}
