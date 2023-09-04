//------------------------------------------------------------------------------
// <copyright file="CommonMatrixUtils.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample.Common
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A collection of matrix utilities for neural network development.
    /// </summary>
    public static class CommonMatrixUtils
    {
        private static readonly double MinClipValue = 1E-15;

        [ThreadStatic]
        private static Random random;

        /// <summary>
        /// Gets a random number generator.
        /// </summary>
        public static Random Random => random ?? (random = new Random((int)((1 + Thread.CurrentThread.ManagedThreadId) * DateTime.UtcNow.Ticks)));

        /// <summary>
        /// Creates an empty matrix of the given size.
        /// </summary>
        /// <param name="numRows">The number of rows.</param>
        /// <param name="numCols">The number of columns.</param>
        /// <returns>An empty matrix.</returns>
        public static Matrix InitializeZeroMatrix(int numRows, int numCols)
        {
            return new Matrix(numRows, numCols);
        }

        /// <summary>
        /// Creates an empty matrix of the given size.
        /// </summary>
        /// <param name="numLayers">The number of lauers.</param>
        /// <param name="numRows">The number of rows.</param>
        /// <param name="numCols">The number of columns.</param>
        /// <returns>An empty matrix.</returns>
        public static Matrix[] InitializeZeroMatrix(int numLayers, int numRows, int numCols)
        {
            Matrix[] matrix = new Matrix[numLayers];
            for (int layerIndex = 0; layerIndex < numLayers; layerIndex++)
            {
                matrix[layerIndex] = new Matrix(numRows, numCols);
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
        public static Matrix[][] InitializeZeroMatrix(int numTimeSteps, int numLayers, int numRows, int numCols)
        {
            Matrix[][] m = new Matrix[numTimeSteps][];
            for (int t = 0; t < numTimeSteps; ++t)
            {
                Matrix[] matrix = new Matrix[numLayers];
                for (int layerIndex = 0; layerIndex < numLayers; layerIndex++)
                {
                    matrix[layerIndex] = new Matrix(numRows, numCols);
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
        public static DeepMatrix[][] ClipGradients(DeepMatrix[][] gradients, double clipValue, double? minValue)
        {
            int dim1 = gradients.Length;
            int dim2 = gradients[0].Length;
            for (int i = 0; i < dim1; ++i)
            {
                for (int j = 0; j < dim2; ++j)
                {
                    gradients[i][j] = ClipGradients(gradients[i][j], clipValue, minValue);
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
        public static DeepMatrix[] ClipGradients(DeepMatrix[] gradients, double clipValue, double? minValue)
        {
            int dim = gradients.Length;
            for (int d = 0; d < dim; ++d)
            {
                gradients[d] = ClipGradients(gradients[d], clipValue, minValue);
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
        public static DeepMatrix ClipGradients(DeepMatrix gradients, double clipValue, double? minValue)
        {
            if (minValue == null)
            {
                minValue = MinClipValue;
            }

            var standardizedMatrix = StandardizedMatrix(gradients);
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

        /// <summary>
        /// Clips gradients to within a certain clip value and applies a minimum threshold value.
        /// </summary>
        /// <param name="gradients">The gradients to clip.</param>
        /// <param name="clipValue">The maximum clipValue in either the positive or negative direction.</param>
        /// <param name="minValue">The minimum threshold value.</param>
        /// <returns>The clipped gradients.</returns>
        public static Matrix ClipGradients(Matrix gradients, double clipValue, double? minValue)
        {
            if (minValue == null)
            {
                minValue = MinClipValue;
            }

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

        /// <summary>
        /// Clips gradients to within a certain clip value and applies a minimum threshold value.
        /// </summary>
        /// <param name="gradients">The gradients to clip.</param>
        /// <param name="clipValue">The maximum clipValue in either the positive or negative direction.</param>
        /// <param name="minValue">The minimum threshold value.</param>
        /// <returns>The clipped gradients.</returns>
        public static Matrix[] ClipGradients(Matrix[] gradients, double clipValue, double? minValue)
        {
            if (minValue == null)
            {
                minValue = MinClipValue;
            }

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
                            gradients[k][i][j] = minValue.Value;
                        }
                        else if (gradients[k][i][j] < 0 && gradients[k][i][j] > -minValue)
                        {
                            gradients[k][i][j] = -minValue.Value;
                        }
                    }
                }
            }

            return gradients;
        }

        /// <summary>
        /// Creates a standardized matrix using the mean and standard deviation.
        /// </summary>
        /// <param name="deepMatrix">The matrix to process.</param>
        /// <returns>The standardized matrix.</returns>
        public static DeepMatrix StandardizedMatrix(DeepMatrix deepMatrix)
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

        /// <summary>
        /// Creates a standardized matrix using the mean and standard deviation.
        /// </summary>
        /// <param name="matrix">The matrix to process.</param>
        /// <returns>The standardized matrix.</returns>
        public static Matrix StandardizedMatrix(Matrix matrix)
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

        /// <summary>
        /// Creates a 2-D array from the specified matrices.
        /// </summary>
        /// <param name="matrices">The matrices.</param>
        /// <returns>The 2-D array.</returns>
        public static Matrix To2DArray(Matrix[] matrices)
        {
            int numMatrices = matrices.Length;
            Matrix matrix = new Matrix(numMatrices, 1);
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
        public static double[] To1DArray(Matrix[] matrices)
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
        public static void SetInPlace(Matrix[] matrices, Matrix[] value)
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
        /// Sets the following deep matrix to the specified values.
        /// </summary>
        /// <param name="matrix">The matrices to replace.</param>
        /// <param name="value">The values to replace the matrix values with.</param>
        public static void SetInPlace(DeepMatrix matrix, DeepMatrix value)
        {
            int numMatrices = matrix.Depth;
            int numRows = matrix.Rows;
            int numCols = matrix.Cols;
            for (int i = 0; i < numMatrices; ++i)
            {
                for (int j = 0; j < numRows; ++j)
                {
                    for (int k = 0; k < numCols; ++k)
                    {
                        matrix[i][j][k] = value[i][j][k];
                    }
                }
            }
        }

        /// <summary>
        /// Sets the following matrices to the specified values.
        /// </summary>
        /// <param name="matrices">The matrices to replace.</param>
        /// <param name="value">The values to replace the matrix values with.</param>
        public static void SetInPlace(FourDimensionalMatrix matrices, FourDimensionalMatrix value)
        {
            int numMatrices = matrices.Count;
            for (int i = 0; i < numMatrices; ++i)
            {
                SetInPlace(matrices[i], value[i]);
            }
        }

        /// <summary>
        /// Sets the following matrix to the specified values.
        /// </summary>
        /// <param name="matrix">The matrix to replace.</param>
        /// <param name="value">The values to replace the matrix values with.</param>
        public static void SetInPlace(Matrix matrix, Matrix value)
        {
            int numRows = matrix.Rows;
            int numCols = matrix.Cols;
            for (int j = 0; j < numRows; ++j)
            {
                for (int k = 0; k < numCols; ++k)
                {
                    matrix[j][k] = value[j][k];
                }
            }
        }

        /// <summary>
        /// Calculates whether the deep matrix is all zeroes.
        /// </summary>
        /// <param name="matrix">The deep matrix.</param>
        /// <returns>A value.</returns>
        public static bool IsAllZeroes(DeepMatrix matrix)
        {
            int numDepth = matrix.Depth;
            int numRows = matrix.Rows;
            int numCols = matrix.Cols;
            for (int i = 0; i < numDepth; ++i)
            {
                for (int j = 0; j < numRows; ++j)
                {
                    for (int k = 0; k < numCols; ++k)
                    {
                        if (matrix[i][j][k] != 0.0d)
                        {
                            return false;
                        }
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Calculates whether the matrix is all zeroes.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        /// <returns>A value.</returns>
        public static bool IsAllZeroes(Matrix matrix)
        {
            int numRows = matrix.Rows;
            int numCols = matrix.Cols;
            for (int j = 0; j < numRows; ++j)
            {
                for (int k = 0; k < numCols; ++k)
                {
                    if (matrix[j][k] != 0.0d)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Clears the following 2-D matrices.
        /// </summary>
        /// <param name="matrices">The 2-D matrices to clear.</param>
        public static void ClearMatrices(Matrix[] matrices)
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
        /// Clears the following 5-D deep matrices.
        /// </summary>
        /// <param name="matrices">The 5-D deep matrices to clear.</param>
        public static void ClearMatrices(DeepMatrix[][] matrices)
        {
            int numMatrices = matrices.Length;

            // Parallelize the outer loop
            Parallel.For(0, numMatrices, i =>
            {
                ClearMatrices(matrices[i]);
            });
        }

        /// <summary>
        /// Clears the following 4-D deep matrices.
        /// </summary>
        /// <param name="matrices">The 4-D deep matrices to clear.</param>
        public static void ClearMatrices(DeepMatrix[] matrices)
        {
            int numMatrices = matrices.Length;

            // Parallelize the outer loop
            Parallel.For(0, numMatrices, i =>
            {
                ClearMatrices(matrices[i].ToArray());
            });
        }

        /// <summary>
        /// The Frobenius norm of a matrix: the square root of the sum of the absolute squares of its elements.
        /// </summary>
        /// <param name="weightMatrix">The weight matrix to calculate.</param>
        /// <returns>The frobenius norm.</returns>
        public static double FrobeniusNorm(Matrix weightMatrix)
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
        public static Matrix HadamardProduct(Matrix matrixA, Matrix matrixB)
        {
            // Check if the dimensions of the matrices match
            int rows = matrixA.Length;
            int cols = matrixA[0].Length;
            if (rows != matrixB.Length || cols != matrixB[0].Length)
            {
                throw new ArgumentException("Matrices must have the same dimensions.");
            }

            // Perform element-wise multiplication
            var result = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++)
            {
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
        public static Matrix MatrixAdd(Matrix a, Matrix b)
        {
            int numRows = a.Length;
            int numCols = a[0].Length;
            Matrix result = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
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
        public static Matrix ScalarMultiply(double scalar, Matrix matrix)
        {
            int numRows = matrix.Length;
            int numCols = matrix[0].Length;

            Matrix result = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
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
        public static Matrix InitializeRandomMatrixWithXavierInitialization(int numRows, int numCols)
        {
            Matrix matrix = new Matrix(numRows, numCols);

            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    matrix[i][j] = ((Random.NextDouble() * 2) - 1) * Math.Sqrt(6.0 / (numRows + numCols));
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
        public static Matrix[] InitializeRandomMatrixWithXavierInitialization(int numLayers, int numRows, int numCols)
        {
            Matrix[] matrix = new Matrix[numLayers];

            Parallel.For(0, numLayers, layerIndex =>
            {
                matrix[layerIndex] = new Matrix(numRows, numCols);
                for (int i = 0; i < numRows; i++)
                {
                    for (int j = 0; j < numCols; j++)
                    {
                        matrix[layerIndex][i][j] = ((Random.NextDouble() * 2) - 1) * Math.Sqrt(6.0 / (numRows + numCols));
                    }
                }
            });

            return matrix;
        }

        /// <summary>
        /// Initialize random matrix with Xavier initialization using the appropriate dimensions.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numFilters">The number of filters.</param>
        /// <param name="depth">The depth.</param>
        /// <param name="numRows">The number of rows.</param>
        /// <param name="numCols">The number of columns.</param>
        /// <returns>The initialized random matrix.</returns>
        public static DeepMatrix[][] InitializeRandomMatrixWithXavierInitialization(int numLayers, int numFilters, int depth, int numRows, int numCols)
        {
            DeepMatrix[][] matrix = new DeepMatrix[numLayers][];

            for (int layerIndex = 0; layerIndex < numLayers; ++layerIndex)
            {
                matrix[layerIndex] = new DeepMatrix[numFilters];
                for (int f = 0; f < numFilters; ++f)
                {
                    matrix[layerIndex][f] = new DeepMatrix(depth, numRows, numCols);
                    matrix[layerIndex][f].Initialize(InitializationType.Xavier);
                }
            }

            return matrix;
        }

        /// <summary>
        /// Initialize random matrix with Xavier initialization using the appropriate dimensions.
        /// </summary>
        /// <param name="numFilters">The number of filters.</param>
        /// <param name="depth">The depth.</param>
        /// <param name="numRows">The number of rows.</param>
        /// <param name="numCols">The number of columns.</param>
        /// <returns>The initialized random matrix.</returns>
        public static DeepMatrix[] InitializeRandomMatrixWithXavierInitialization(int numFilters, int depth, int numRows, int numCols)
        {
            DeepMatrix[] matrix = new DeepMatrix[numFilters];
            for (int f = 0; f < numFilters; ++f)
            {
                matrix[f] = new DeepMatrix(depth, numRows, numCols);
                matrix[f].Initialize(InitializationType.Xavier);
            }

            return matrix;
        }

        /// <summary>
        /// Switch the first two dimensions of the deep matrix array.
        /// </summary>
        /// <param name="deepMatrixArray">The deep matrix array.</param>
        /// <returns>The deep matrix array switched.</returns>
        public static DeepMatrix[] SwitchFirstTwoDimensions(DeepMatrix[] deepMatrixArray)
        {
            DeepMatrix[] switched = new DeepMatrix[deepMatrixArray[0].Depth];
            for (int i = 0; i < deepMatrixArray.Length; ++i)
            {
                for (int j = 0; j < deepMatrixArray[0].Depth; ++j)
                {
                    if (switched[j] == null)
                    {
                        switched[j] = new DeepMatrix(deepMatrixArray.Length, 1, 1);
                    }

                    switched[j][i] = deepMatrixArray[i][j];
                }
            }

            return switched;
        }

        /// <summary>
        /// Sets the following matrices to the specified values.
        /// </summary>
        /// <param name="matrices">The matrices to replace.</param>
        /// <param name="value">The values to replace the matrix values with.</param>
        public static void SetInPlaceReplace(FourDimensionalMatrix matrices, FourDimensionalMatrix value)
        {
            int numMatrices = matrices.Count;
            for (int i = 0; i < numMatrices; ++i)
            {
                matrices[i].Replace(value[i].ToArray());
            }
        }

        /// <summary>
        /// Sets the following matrices to the specified values.
        /// </summary>
        /// <param name="matrices">The matrices to replace.</param>
        /// <param name="value">The values to replace the matrix values with.</param>
        public static void SetInPlaceReplace(DeepMatrix matrices, DeepMatrix value)
        {
            matrices.Replace(value.ToArray());
        }

        /// <summary>
        /// Sets the following matrix to the specified values.
        /// </summary>
        /// <param name="matrix">The matrix to replace.</param>
        /// <param name="value">The values to replace the matrix values with.</param>
        public static void SetInPlaceReplace(Matrix matrix, Matrix value)
        {
            matrix.Replace(value.ToArray());
        }
    }
}
