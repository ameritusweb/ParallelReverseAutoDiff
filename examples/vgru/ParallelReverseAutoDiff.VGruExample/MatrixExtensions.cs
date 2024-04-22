// ------------------------------------------------------------------------------
// <copyright file="MatrixExtensions.cs" author="ameritusweb" date="12/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Matrix extensions.
    /// </summary>
    public static class MatrixExtensions
    {
        /// <summary>
        /// Append a matrix to another matrix.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        /// <param name="other">The other.</param>
        /// <param name="direction">The direction.</param>
        /// <returns>The result.</returns>
        /// <exception cref="ArgumentOutOfRangeException">An exception.</exception>
        public static Matrix Append(this Matrix matrix, Matrix other, AppendDirection direction)
        {
            switch (direction)
            {
                case AppendDirection.Right:
                    return AppendRight(matrix, other);
                case AppendDirection.Left:
                    return AppendLeft(matrix, other);
                case AppendDirection.Up:
                    return AppendUp(matrix, other);
                case AppendDirection.Down:
                    return AppendDown(matrix, other);
                default:
                    throw new ArgumentOutOfRangeException(nameof(direction), "Invalid append direction");
            }
        }

        private static Matrix AppendRight(Matrix matrix, Matrix other)
        {
            if (matrix.Rows != other.Rows)
            {
                throw new ArgumentException("Rows must match to append right.");
            }

            Matrix result = new Matrix(matrix.Rows, matrix.Cols + other.Cols);
            CopyInto(matrix, result, 0, 0);
            CopyInto(other, result, 0, matrix.Cols);
            return result;
        }

        private static Matrix AppendLeft(Matrix matrix, Matrix other)
        {
            return AppendRight(other, matrix);
        }

        private static Matrix AppendUp(Matrix matrix, Matrix other)
        {
            if (matrix.Cols != other.Cols)
            {
                throw new ArgumentException("Columns must match to append upwards.");
            }

            Matrix result = new Matrix(matrix.Rows + other.Rows, matrix.Cols);
            CopyInto(other, result, 0, 0);
            CopyInto(matrix, result, other.Rows, 0);
            return result;
        }

        private static Matrix AppendDown(Matrix matrix, Matrix other)
        {
            return AppendUp(other, matrix);
        }

        private static void CopyInto(Matrix source, Matrix target, int startRow, int startCol)
        {
            for (int i = 0; i < source.Rows; i++)
            {
                for (int j = 0; j < source.Cols; j++)
                {
                    target[startRow + i, startCol + j] = source[i, j];
                }
            }
        }
    }
}
