//------------------------------------------------------------------------------
// <copyright file="VectorToMatrix.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Provides the ability to convert a vector to a matrix.
    /// </summary>
    public class VectorToMatrix
    {
        /// <summary>
        /// Create a line matrix.
        /// </summary>
        /// <param name="angle">The angle.</param>
        /// <param name="size">The size.</param>
        /// <returns>The matrix.</returns>
        public static Matrix CreateLine(double angle, int size = 11)
        {
            Matrix matrix = new Matrix(size, size);

            // Initialize the matrix background to 0.1
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    matrix[i, j] = 0.1d;
                }
            }

            // Calculate the start and end points of the line
            int centerX = size / 2;
            int centerY = size / 2;
            int endX = (int)(centerX + ((size - 1) * Math.Cos(angle)));
            int endY = (int)(centerY + ((size - 1) * Math.Sin(angle)));

            // Draw the line using Bresenham's line algorithm
            DrawLine(matrix, centerX, centerY, endX, endY, size);

            return matrix;
        }

        private static void DrawLine(Matrix matrix, int x0, int y0, int x1, int y1, int size)
        {
            int dx = Math.Abs(x1 - x0);
            int dy = -Math.Abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1;
            int sy = y0 < y1 ? 1 : -1;
            int err = dx + dy, e2;

            while (true)
            {
                if (x0 >= 0 && x0 < size && y0 >= 0 && y0 < size)
                {
                    matrix[y0, x0] = 0.9; // Set the line part to 0.9
                }

                if (x0 == x1 && y0 == y1)
                {
                    break;
                }

                e2 = 2 * err;
                if (e2 >= dy)
                {
                    err += dy;
                    x0 += sx;
                }

                if (e2 <= dx)
                {
                    err += dx;
                    y0 += sy;
                }
            }
        }
    }
}
