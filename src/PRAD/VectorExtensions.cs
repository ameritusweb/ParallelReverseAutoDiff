//------------------------------------------------------------------------------
// <copyright file="VectorExtensions.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Numerics;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Vector Extensions.
    /// </summary>
    public static class VectorExtensions
    {
        /// <summary>
        /// Convert to Cartesian.
        /// </summary>
        /// <param name="v">The vector to convert.</param>
        /// <returns>The vector in Cartesian.</returns>
        public static Vector2 ToCartesian(this Vector2 v)
        {
            return new Vector2((float)(v.X * Math.Cos(v.Y)), (float)(v.X * Math.Cos(v.Y)));
        }

        /// <summary>
        /// Subtract two polar vectors.
        /// </summary>
        /// <param name="v1">The first polar vector.</param>
        /// <param name="v2">The second polar vector.</param>
        /// <returns>THe result.</returns>
        public static Vector2 Sub(this Vector2 v1, Vector2 v2)
        {
            var c1 = v1.ToCartesian();
            var c2 = v2.ToCartesian();
            return new Vector2(c1.X - c2.X, c1.Y - c2.Y);
        }

        /// <summary>
        /// Creates an interleaved tensor.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        /// <returns>An interleaved tensor.</returns>
        public static Tensor ToInterleavedTensor(this Vector2[][] matrix)
        {
            int rows = matrix.Length;
            int cols = matrix[0].Length;
            var output = new Tensor(new int[] { rows, cols * 2 });
            var data = PradTools.AllocateArray(rows * cols * 2);
            int index = 0;
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    data[index++] = matrix[i][j].X;
                    data[cols + index++] = matrix[i][j].Y;
                }

                index += cols;
            }

            return new Tensor(new int[] { rows, cols * 2 }, data);
        }
    }
}
