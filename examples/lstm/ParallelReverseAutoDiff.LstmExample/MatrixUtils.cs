//------------------------------------------------------------------------------
// <copyright file="MatrixUtils.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.LstmExample
{
    /// <summary>
    /// A collection of matrix utilities for neural network development.
    /// </summary>
    public static class MatrixUtils
    {
        /// <summary>
        /// Creates an empty matrix of the given size.
        /// </summary>
        /// <param name="numRows">The number of rows.</param>
        /// <param name="numCols">The number of columns.</param>
        /// <returns>An empty matrix.</returns>
        public static double[][] InitializeZeroMatrix(int numRows, int numCols)
        {
            double[][] matrix = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                matrix[i] = new double[numCols];
            }

            return matrix;
        }

        /// <summary>
        /// Creates an empty matrix of the given size.
        /// </summary>
        /// <param name="numLayers">The number of lauers.</param>
        /// <param name="numRows">The number of rows.</param>
        /// <param name="numCols">The number of columns.</param>
        /// <returns>An empty matrix.</returns>
        public static double[][][] InitializeZeroMatrix(int numLayers, int numRows, int numCols)
        {
            double[][][] matrix = new double[numLayers][][];
            for (int layerIndex = 0; layerIndex < numLayers; layerIndex++)
            {
                matrix[layerIndex] = new double[numRows][];
                for (int i = 0; i < numRows; i++)
                {
                    matrix[layerIndex][i] = new double[numCols];
                }
            }

            return matrix;
        }

        /// <summary>
        /// Creates an empty matrix of the given size.
        /// </summary>
        /// <param name="numTimeSteps">The number of time steps.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numRows">The number of rows.</param>
        /// <param name="numCols">The number of columns.</param>
        /// <returns>An empty matrix.</returns>
        public static double[][][][] InitializeZeroMatrix(int numTimeSteps, int numLayers, int numRows, int numCols)
        {
            double[][][][] m = new double[numTimeSteps][][][];
            for (int t = 0; t < numTimeSteps; ++t)
            {
                double[][][] matrix = new double[numLayers][][];
                for (int layerIndex = 0; layerIndex < numLayers; layerIndex++)
                {
                    matrix[layerIndex] = new double[numRows][];
                    for (int i = 0; i < numRows; i++)
                    {
                        matrix[layerIndex][i] = new double[numCols];
                    }
                }

                m[t] = matrix;
            }

            return m;
        }

        /// <summary>
        /// Clips gradients to within a certain clip value and applies a minimum threshold value.
        /// </summary>
        /// <param name="gradients">The gradients to clip.</param>
        /// <param name="clipValue">The maximum clipValue in either the positive or negative direction.</param>
        /// <param name="minValue">The minimum threshold value.</param>
        /// <returns>The clipped gradients.</returns>
        public static double[][] ClipGradients(double[][] gradients, double clipValue, double minValue = 1E-6)
        {
            var standardizedMatrix = StandardizedMatrix(gradients);
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
                        gradients[i][j] = minValue;
                    }
                    else if (gradients[i][j] < 0 && gradients[i][j] > -minValue)
                    {
                        gradients[i][j] = -minValue;
                    }
                }
            }

            return gradients;
        }

        /// <summary>
        /// Clips gradients to within a certain clip value and applies a minimum threshold value.
        /// </summary>
        /// <param name="gradients">The gradients to clip.</param>
        /// <param name="clipValue">The maximum clipValue in either the positive or negative direction.</param>
        /// <param name="minValue">The minimum threshold value.</param>
        /// <returns>The clipped gradients.</returns>
        public static double[][][] ClipGradients(double[][][] gradients, double clipValue, double minValue = 1E-6)
        {
            int numMatrices = gradients.Length;

            for (int k = 0; k < numMatrices; k++)
            {
                int numRows = gradients[k].Length;
                int numCols = gradients[k][0].Length;
                var standardizedMatrix = StandardizedMatrix(gradients[k]);

                for (int i = 0; i < numRows; i++)
                {
                    for (int j = 0; j < numCols; j++)
                    {
                        var value = Math.Min(clipValue, 1 + Math.Abs(standardizedMatrix[i][j]));

                        // Clip the gradient values
                        if (gradients[k][i][j] > value)
                        {
                            gradients[k][i][j] = value;
                        }
                        else if (gradients[k][i][j] < -value)
                        {
                            gradients[k][i][j] = -value;
                        }

                        // Apply the minimum threshold value
                        if (gradients[k][i][j] > 0 && gradients[k][i][j] < minValue)
                        {
                            gradients[k][i][j] = minValue;
                        }
                        else if (gradients[k][i][j] < 0 && gradients[k][i][j] > -minValue)
                        {
                            gradients[k][i][j] = -minValue;
                        }
                    }
                }
            }

            return gradients;
        }

        /// <summary>
        /// Creates a standardized matrix using the mean and standard deviation.
        /// </summary>
        /// <param name="matrix">The matrix to process.</param>
        /// <returns>The standardized matrix.</returns>
        public static double[][] StandardizedMatrix(double[][] matrix)
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
            double[][] standardizedMatrix = new double[matrix.Length][];
            for (int i = 0; i < matrix.Length; i++)
            {
                standardizedMatrix[i] = new double[matrix[i].Length];
                for (int j = 0; j < matrix[i].Length; j++)
                {
                    standardizedMatrix[i][j] = (matrix[i][j] - mean) / stdDev;
                }
            }

            return standardizedMatrix;
        }

        /// <summary>
        /// Creates a 2-D array from the specified matrices.
        /// </summary>
        /// <param name="matrices">The matrices.</param>
        /// <returns>The 2-D array.</returns>
        public static double[][] To2DArray(double[][][] matrices)
        {
            int numMatrices = matrices.Length;
            double[][] matrix = new double[numMatrices][];
            for (int i = 0; i < numMatrices; ++i)
            {
                matrix[i] = matrices[i][0];
            }

            return matrix;
        }

        /// <summary>
        /// Creates a 1-D array from the specified matrices.
        /// </summary>
        /// <param name="matrices">The matrices.</param>
        /// <returns>The 1-D array.</returns>
        public static double[] To1DArray(double[][][] matrices)
        {
            int numMatrices = matrices.Length;
            double[] array = new double[numMatrices];
            for (int i = 0; i < numMatrices; ++i)
            {
                array[i] = matrices[i][0][0];
            }

            return array;
        }

        /// <summary>
        /// Sets the following matrices to the specified values.
        /// </summary>
        /// <param name="matrices">The matrices to replace.</param>
        /// <param name="value">The values to replace the matrix values with.</param>
        public static void SetInPlace(double[][][] matrices, double[][][] value)
        {
            int numMatrices = matrices.Length;
            int numRows = matrices[0].Length;
            int numCols = matrices[0][0].Length;
            for (int i = 0; i < numMatrices; ++i)
            {
                for (int j = 0; j < numRows; ++j)
                {
                    for (int k = 0; k < numCols; ++k)
                    {
                        matrices[i][j][k] = value[i][j][k];
                    }
                }
            }
        }

        /// <summary>
        /// Clears the following 4-D matrices.
        /// </summary>
        /// <param name="matrices">The 4-D matrices to clear.</param>
        public static void ClearArrays4D(double[][][][][] matrices)
        {
            int numMatrices = matrices.Length;
            int numTimesteps = matrices[0].Length;
            int numLayers = matrices[0][0].Length;

            // Parallelize the outer loop
            Parallel.For(0, numMatrices, i =>
            {
                for (int j = 0; j < numTimesteps; ++j)
                {
                    for (int k = 0; k < numLayers; ++k)
                    {
                        int numRows = matrices[i][j][k].Length;
                        int numCols = matrices[i][j][k][0].Length;
                        for (int l = 0; l < numRows; ++l)
                        {
                            matrices[i][j][k][l] = new double[numCols];
                        }
                    }
                }
            });
        }

        /// <summary>
        /// Clears the following 3-D matrices.
        /// </summary>
        /// <param name="matrices">The 3-D matrices to clear.</param>
        public static void ClearArrays3D(double[][][][] matrices)
        {
            int numMatrices = matrices.Length;
            int numLayers = matrices[0].Length;

            // Parallelize the outer loop
            Parallel.For(0, numMatrices, i =>
            {
                for (int j = 0; j < numLayers; ++j)
                {
                    int numRows = matrices[i][j].Length;
                    int numCols = matrices[i][j][0].Length;
                    for (int k = 0; k < numRows; ++k)
                    {
                        matrices[i][j][k] = new double[numCols];
                    }
                }
            });
        }

        /// <summary>
        /// Clears the following 2-D matrices.
        /// </summary>
        /// <param name="matrices">The 2-D matrices to clear.</param>
        public static void ClearArrays2D(double[][][] matrices)
        {
            int numMatrices = matrices.Length;

            // Parallelize the outer loop
            Parallel.For(0, numMatrices, i =>
            {
                int numRows = matrices[i].Length;
                int numCols = matrices[i][0].Length;
                for (int j = 0; j < numRows; ++j)
                {
                    matrices[i][j] = new double[numCols];
                }
            });
        }

        /// <summary>
        /// The Frobenius norm of a matrix: the square root of the sum of the absolute squares of its elements.
        /// </summary>
        /// <param name="weightMatrix">The weight matrix to calculate.</param>
        /// <returns>The frobenius norm.</returns>
        public static double FrobeniusNorm(double[][] weightMatrix)
        {
            double sum = 0.0;

            for (int i = 0; i < weightMatrix.Length; i++)
            {
                for (int j = 0; j < weightMatrix[i].Length; j++)
                {
                    sum += weightMatrix[i][j] * weightMatrix[i][j];
                }
            }

            return Math.Sqrt(sum);
        }

        /// <summary>
        /// Calculates the reduction factor for the learning rate.
        /// </summary>
        /// <param name="frobeniusNorm">The frobenius norm of a matrix.</param>
        /// <param name="maxNorm">The max norm.</param>
        /// <param name="minFactor">The minimum learning rate reduction factor.</param>
        /// <returns>The learning rate reduction factor.</returns>
        public static double LearningRateReductionFactor(double frobeniusNorm, double maxNorm, double minFactor)
        {
            if (frobeniusNorm <= maxNorm)
            {
                return 1.0;
            }
            else
            {
                double reductionFactor = Math.Pow(maxNorm / frobeniusNorm, 3);
                return Math.Max(reductionFactor, minFactor);
            }
        }

        /// <summary>
        /// The element-wise Hadamard product of two matrices.
        /// </summary>
        /// <param name="matrixA">The first matrix.</param>
        /// <param name="matrixB">The second matrix.</param>
        /// <returns>The resultant matrix.</returns>
        public static double[][] HadamardProduct(double[][] matrixA, double[][] matrixB)
        {
            // Check if the dimensions of the matrices match
            int rows = matrixA.Length;
            int cols = matrixA[0].Length;
            if (rows != matrixB.Length || cols != matrixB[0].Length)
            {
                throw new ArgumentException("Matrices must have the same dimensions.");
            }

            // Perform element-wise multiplication
            var result = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                result[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    result[i][j] = matrixA[i][j] * matrixB[i][j];
                }
            }

            return result;
        }

        /// <summary>
        /// Add two matrices together.
        /// </summary>
        /// <param name="a">Matrix A.</param>
        /// <param name="b">Matrix B.</param>
        /// <returns>The resultant matrix.</returns>
        public static double[][] MatrixAdd(double[][] a, double[][] b)
        {
            int numRows = a.Length;
            int numCols = a[0].Length;
            double[][] result = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                result[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    result[i][j] = a[i][j] + b[i][j];
                }
            }

            return result;
        }

        /// <summary>
        /// Multiply a matrix by a scalar.
        /// </summary>
        /// <param name="scalar">The scalar to multiply.</param>
        /// <param name="matrix">The matrix.</param>
        /// <returns>The resultant matrix.</returns>
        public static double[][] ScalarMultiply(double scalar, double[][] matrix)
        {
            int numRows = matrix.Length;
            int numCols = matrix[0].Length;

            double[][] result = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                result[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    result[i][j] = scalar * matrix[i][j];
                }
            }

            return result;
        }

        /// <summary>
        /// Initialize random matrix with Xavier initialization using the appropriate dimensions.
        /// </summary>
        /// <param name="numRows">The number of rows.</param>
        /// <param name="numCols">The number of columns.</param>
        /// <returns>The initialized random matrix.</returns>
        public static double[][] InitializeRandomMatrixWithXavierInitialization(int numRows, int numCols)
        {
            double[][] matrix = new double[numRows][];
            var randomFunc = () => new Random(Guid.NewGuid().GetHashCode());
            var localRandom = new ThreadLocal<Random>(randomFunc);
            var rand = localRandom == null || localRandom.Value == null ? randomFunc() : localRandom.Value;

            Parallel.For(0, numRows, i =>
            {
                matrix[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    matrix[i][j] = ((rand.NextDouble() * 2) - 1) * Math.Sqrt(6.0 / (numRows + numCols));
                }
            });

            return matrix;
        }

        /// <summary>
        /// Initialize random matrix with Xavier initialization using the appropriate dimensions.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numRows">The number of rows.</param>
        /// <param name="numCols">The number of columns.</param>
        /// <returns>The initialized random matrix.</returns>
        public static double[][][] InitializeRandomMatrixWithXavierInitialization(int numLayers, int numRows, int numCols)
        {
            double[][][] matrix = new double[numLayers][][];

            var randomFunc = () => new Random(Guid.NewGuid().GetHashCode());
            var localRandom = new ThreadLocal<Random>(randomFunc);
            var rand = localRandom == null || localRandom.Value == null ? randomFunc() : localRandom.Value;

            Parallel.For(0, numLayers, layerIndex =>
            {
                matrix[layerIndex] = new double[numRows][];
                for (int i = 0; i < numRows; i++)
                {
                    matrix[layerIndex][i] = new double[numCols];
                    for (int j = 0; j < numCols; j++)
                    {
                        matrix[layerIndex][i][j] = ((rand.NextDouble() * 2) - 1) * Math.Sqrt(6.0 / (numRows + numCols));
                    }
                }
            });

            return matrix;
        }
    }
}
