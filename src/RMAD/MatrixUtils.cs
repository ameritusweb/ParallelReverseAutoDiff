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
    }
}
