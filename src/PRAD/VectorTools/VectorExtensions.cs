//------------------------------------------------------------------------------
// <copyright file="VectorExtensions.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.VectorTools
{
    using System;
    using System.Numerics;
    using ParallelReverseAutoDiff.PRAD;
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
        /// Convert to Polar.
        /// </summary>
        /// <param name="v">The vector to convert.</param>
        /// <returns>The vector in Polar.</returns>
        public static (double Mag, double Ang) ToPolar(this Vector2 v)
        {
            var mag = Math.Sqrt(Math.Pow(v.X, 2d) + Math.Pow(v.Y, 2d));
            var ang = Math.Atan2(v.Y, v.X);
            return (mag, ang);
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
        /// Rotate a vector.
        /// </summary>
        /// <param name="vector">The vector to rotate.</param>
        /// <param name="angle">The angle to rotate by.</param>
        /// <param name="origin">Thw origin point.</param>
        /// <returns>The rotated vector.</returns>
        public static Vector2 Rotate(this Vector2 vector, float angle, Vector2? origin = null)
        {
            if (origin.HasValue)
            {
                vector -= origin.Value;
            }

            float cos = (float)Math.Cos(angle);
            float sin = (float)Math.Sin(angle);
            Vector2 rotated = new Vector2(
                (vector.X * cos) - (vector.Y * sin),
                (vector.X * sin) + (vector.Y * cos));

            if (origin.HasValue)
            {
                rotated += origin.Value;
            }

            return rotated;
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
