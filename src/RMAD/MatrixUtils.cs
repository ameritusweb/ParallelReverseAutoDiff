//------------------------------------------------------------------------------
// <copyright file="MatrixUtils.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading;
    using System.Threading.Tasks;

    /// <summary>
    /// Matrix utilities for reverse mode automatic differentiation.
    /// </summary>
    public static class MatrixUtils
    {
        [ThreadStatic]
        private static Random random;

        /// <summary>
        /// Gets a random number generator for the current thread.
        /// </summary>
        public static Random Random => random ?? (random = new Random((int)((1 + Thread.CurrentThread.ManagedThreadId) * DateTime.UtcNow.Ticks)));

        /// <summary>
        /// Flattens a matrix into a 1D array.
        /// </summary>
        /// <param name="matrix">The matrix to flatten.</param>
        /// <returns>The 1-D array.</returns>
        public static double[] FlattenMatrix(Matrix matrix)
        {
            int rows = matrix.Length;
            int cols = matrix[0].Length;
            double[] flat = new double[rows * cols];
            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    flat[(i * cols) + j] = matrix[i][j];
                }
            });
            return flat;
        }

        /// <summary>
        /// Reshape a 1D array into a matrix.
        /// </summary>
        /// <param name="flat">The 1-D array.</param>
        /// <param name="rows">The number of rows.</param>
        /// <param name="cols">The number of columns.</param>
        /// <returns>A reshaped matrix.</returns>
        public static Matrix ReshapeMatrix(double[] flat, int rows, int cols)
        {
            Matrix matrix = new Matrix(rows, cols);
            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i][j] = flat[(i * cols) + j];
                }
            });
            return matrix;
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
            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i][j] = matrixA[i][j] * matrixB[i][j];
                }
            });

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
            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    result[i][j] = a[i][j] + b[i][j];
                }
            });

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
            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    result[i][j] = scalar * matrix[i][j];
                }
            });

            return result;
        }
    }
}
