//------------------------------------------------------------------------------
// <copyright file="GradientClipper.cs" author="ameritusweb" date="5/7/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;

    /// <summary>
    /// A gradient clipper.
    /// </summary>
    public class GradientClipper
    {
        private NeuralNetwork network;

        /// <summary>
        /// Initializes a new instance of the <see cref="GradientClipper"/> class.
        /// </summary>
        /// <param name="network">The neural network.</param>
        public GradientClipper(NeuralNetwork network)
        {
            this.network = network;
        }

        /// <summary>
        /// Clip the layers.
        /// </summary>
        /// <param name="layers">The layers to clip.</param>
        public void Clip(IModelLayer[] layers)
        {
            Parallel.For(0, layers.Length, i =>
            {
                var layer = layers[i];
                var identifiers = layer.Identifiers;
                for (int j = 0; j < identifiers.Count; ++j)
                {
                    var identifier = identifiers[j];
                    var gradient = layer[identifier, ModelElementType.Gradient];
                    var dimensions = layer.Dimensions(identifier) ?? throw new InvalidOperationException("Dimensions cannot be null.");
                    switch (dimensions.Length)
                    {
                        case 2:
                            {
                                var gradientMatrix = gradient as Matrix ?? throw new InvalidOperationException("Gradient cannot be null.");
                                this.ClipGradients(gradientMatrix, this.network.Parameters.ClipValue, null);
                                break;
                            }

                        case 3:
                            {
                                var gradientMatrix = gradient as DeepMatrix ?? throw new InvalidOperationException("Gradient cannot be null.");
                                this.ClipGradients(gradientMatrix, this.network.Parameters.ClipValue, null);
                                break;
                            }

                        case 4:
                            {
                                var gradientMatrix = gradient as DeepMatrix[] ?? throw new InvalidOperationException("Gradient cannot be null.");
                                for (int d = 0; d < dimensions[0]; ++d)
                                {
                                    this.ClipGradients(gradientMatrix[d], this.network.Parameters.ClipValue, null);
                                }

                                break;
                            }
                    }
                }
            });
        }

        private DeepMatrix ClipGradients(DeepMatrix gradients, double clipValue, double? minValue)
        {
            if (minValue == null)
            {
                minValue = this.network.Parameters.MinimumClipValue;
            }

            var standardizedMatrix = this.StandardizedMatrix(gradients);
            int depth = gradients.Depth;
            int numRows = gradients.Rows;
            int numCols = gradients.Cols;
            Parallel.For(0, depth, d =>
            {
                for (int i = 0; i < numRows; i++)
                {
                    for (int j = 0; j < numCols; j++)
                    {
                        var value = Math.Min(clipValue, 1 + Math.Abs(standardizedMatrix[d, i, j]));

                        // Clip the gradient values
                        if (gradients[d, i, j] > value)
                        {
                            gradients[d, i, j] = value;
                        }
                        else if (gradients[d, i, j] < -value)
                        {
                            gradients[d, i, j] = -value;
                        }

                        // Apply the minimum threshold value
                        if (gradients[d, i, j] > 0 && gradients[d, i, j] < minValue)
                        {
                            gradients[d, i, j] = minValue.Value;
                        }
                        else if (gradients[d, i, j] < 0 && gradients[d, i, j] > -minValue)
                        {
                            gradients[d, i, j] = -minValue.Value;
                        }
                    }
                }
            });

            return gradients;
        }

        private Matrix ClipGradients(Matrix gradients, double clipValue, double? minValue)
        {
            if (minValue == null)
            {
                minValue = this.network.Parameters.MinimumClipValue;
            }

            var standardizedMatrix = this.StandardizedMatrix(gradients);
            int numRows = gradients.Length;
            int numCols = gradients[0].Length;
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    var value = Math.Min(clipValue, 1 + Math.Abs(standardizedMatrix[i][j]));

                    // Clip the gradient values
                    if (gradients[i][j] > value)
                    {
                        gradients[i][j] = value;
                    }
                    else if (gradients[i][j] < -value)
                    {
                        gradients[i][j] = -value;
                    }

                    // Apply the minimum threshold value
                    if (gradients[i][j] > 0 && gradients[i][j] < minValue)
                    {
                        gradients[i][j] = minValue.Value;
                    }
                    else if (gradients[i][j] < 0 && gradients[i][j] > -minValue)
                    {
                        gradients[i][j] = -minValue.Value;
                    }
                }
            }

            return gradients;
        }

        private DeepMatrix StandardizedMatrix(DeepMatrix deepMatrix)
        {
            // Calculate the mean
            double sum = 0;
            int count = 0;
            foreach (var matrix in deepMatrix)
            {
                foreach (double[] row in matrix)
                {
                    foreach (double value in row)
                    {
                        sum += value;
                        count++;
                    }
                }
            }

            double mean = sum / count;

            // Calculate the standard deviation
            double varianceSum = 0;
            foreach (var matrix in deepMatrix)
            {
                foreach (double[] row in matrix)
                {
                    foreach (double value in row)
                    {
                        varianceSum += Math.Pow(value - mean, 2);
                    }
                }
            }

            double stdDev = Math.Sqrt(varianceSum / count);

            // Calculate the standardized matrix
            int depth = deepMatrix.Depth;
            int rows = deepMatrix.Rows;
            int cols = deepMatrix.Cols;
            DeepMatrix standardizedMatrix = new DeepMatrix(depth, rows, cols);
            Parallel.For(0, depth, d =>
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        standardizedMatrix[d, i, j] = (deepMatrix[d, i, j] - mean) / stdDev;
                    }
                }
            });

            return standardizedMatrix;
        }

        private Matrix StandardizedMatrix(Matrix matrix)
        {
            // Calculate the mean
            double sum = 0;
            int count = 0;
            foreach (double[] row in matrix)
            {
                foreach (double value in row)
                {
                    sum += value;
                    count++;
                }
            }

            double mean = sum / count;

            // Calculate the standard deviation
            double varianceSum = 0;
            foreach (double[] row in matrix)
            {
                foreach (double value in row)
                {
                    varianceSum += Math.Pow(value - mean, 2);
                }
            }

            double stdDev = Math.Sqrt(varianceSum / count);

            // Calculate the standardized matrix
            int rows = matrix.Rows;
            int cols = matrix.Cols;
            Matrix standardizedMatrix = new Matrix(rows, cols);
            for (int i = 0; i < matrix.Length; i++)
            {
                for (int j = 0; j < matrix[i].Length; j++)
                {
                    standardizedMatrix[i][j] = (matrix[i][j] - mean) / stdDev;
                }
            }

            return standardizedMatrix;
        }
    }
}
