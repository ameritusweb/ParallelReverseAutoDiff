//------------------------------------------------------------------------------
// <copyright file="TensorAngleExtensions.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;

    /// <summary>
    /// Calculates the co-efficient for squared arclength Euclidean loss.
    /// </summary>
    public static class TensorAngleExtensions
    {
        /// <summary>
        /// Calculates the eo-efficient for squared arclength Euclidean loss.
        /// </summary>
        /// <param name="angles">The angles.</param>
        /// <returns>The co-efficient.</returns>
        /// <exception cref="ArgumentException">Must be 1x2.</exception>
        /// <exception cref="InvalidOperationException">Invalid quadrant.</exception>
        public static (double X, double Y) CalculateCoefficient(this Tensor angles)
        {
            if (angles.Shape[0] != 1 || angles.Shape[1] != 2)
            {
                throw new ArgumentException("Tensor must be a 1x2 tensor containing actual and target angles.");
            }

            double actualAngle = angles[0, 0];
            double targetAngle = angles[0, 1];

            int GetQuadrant(double angleInRadians)
            {
                double normalizedAngle = angleInRadians % (2 * Math.PI);
                if (normalizedAngle < 0)
                {
                    normalizedAngle += 2 * Math.PI;
                }

                if (normalizedAngle < Math.PI / 2)
                {
                    return 1;
                }

                if (normalizedAngle < Math.PI)
                {
                    return 2;
                }

                if (normalizedAngle < 3 * Math.PI / 2)
                {
                    return 3;
                }

                return 4;
            }

            double CalculateOppositeAngle(double angle)
            {
                double oppositeAngle = angle + Math.PI;
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

            int actualQuadrant = GetQuadrant(actualAngle);
            int targetQuadrant = GetQuadrant(targetAngle);
            double oppositeAngle = CalculateOppositeAngle(targetAngle);

            if (targetQuadrant == 1)
            {
                if (actualQuadrant == 1)
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
                else if (actualQuadrant == 2)
                {
                    return (-1, -1); // x decrease, y decrease
                }
                else if (actualQuadrant == 3)
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
                if (actualQuadrant == 1)
                {
                    return (1, -1); // x increase, y decrease
                }
                else if (actualQuadrant == 2)
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
                else if (actualQuadrant == 3)
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
                if (actualQuadrant == 1)
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
                else if (actualQuadrant == 2)
                {
                    return (1, 1); // x increase, y increase
                }
                else if (actualQuadrant == 3)
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
                if (actualQuadrant == 1)
                {
                    return (-1, 1); // x decrease, y increase
                }
                else if (actualQuadrant == 2)
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
                else if (actualQuadrant == 3)
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
    }
}
