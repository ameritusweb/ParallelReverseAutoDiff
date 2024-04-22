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
        /// Concatenate a deep matrix.
        /// </summary>
        /// <param name="deepMatrix">The deep matrix.</param>
        /// <param name="direction">The direction.</param>
        /// <returns>A whole matrix.</returns>
        public static Matrix ConcatItself(this DeepMatrix deepMatrix, AppendDirection direction)
        {
            Matrix result = deepMatrix[0];
            for (int i = 1; i < deepMatrix.Depth; ++i)
            {
                result = result.Append(deepMatrix[i], direction);
            }

            return result;
        }

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
                case AppendDirection.VectorRight:
                    return AppendVectorRight(matrix, other);
                case AppendDirection.VectorLeft:
                    return AppendVectorLeft(matrix, other);
                default:
                    throw new ArgumentOutOfRangeException(nameof(direction), "Invalid append direction");
            }
        }

        private static Matrix AppendVectorRight(Matrix matrix, Matrix other)
        {
            // Ensure rows match for proper appending
            if (matrix.Rows != other.Rows)
            {
                throw new ArgumentException("Rows must match to append vectors right.");
            }

            int matrixHalfCols = matrix.Cols / 2;
            int otherHalfCols = other.Cols / 2;

            Matrix result = new Matrix(matrix.Rows, matrix.Cols + other.Cols);

            // Magnitudes of both matrices appended first
            CopyInto(matrix, result, 0, 0, 0, matrixHalfCols); // Copy magnitudes from first matrix
            CopyInto(other, result, 0, matrixHalfCols, 0, otherHalfCols); // Copy magnitudes from second matrix

            // Angles of both matrices appended second
            CopyInto(matrix, result, 0, matrixHalfCols + otherHalfCols, matrixHalfCols, matrix.Cols); // Copy angles from first matrix
            CopyInto(other, result, 0, matrix.Cols + otherHalfCols, otherHalfCols, other.Cols); // Copy angles from second matrix

            return result;
        }

        private static Matrix AppendVectorLeft(Matrix matrix, Matrix other)
        {
            // Reverse the operation of AppendVectorRight
            return AppendVectorRight(other, matrix);
        }

        private static Matrix AppendRight(Matrix matrix, Matrix other)
        {
            if (matrix.Rows != other.Rows)
            {
                throw new ArgumentException("Rows must match to append right.");
            }

            Matrix result = new Matrix(matrix.Rows, matrix.Cols + other.Cols);
            CopyInto(matrix, result, 0, 0, 0, matrix.Cols); // Copy entire first matrix
            CopyInto(other, result, 0, matrix.Cols, 0, other.Cols); // Copy entire second matrix
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

        private static void CopyInto(Matrix source, Matrix target, int targetStartRow, int targetStartCol, int sourceStartCol = 0, int sourceEndCol = -1)
        {
            // If no end column specified, assume copying the whole row
            if (sourceEndCol == -1)
            {
                sourceEndCol = source.Cols;
            }

            for (int i = 0; i < source.Rows; i++)
            {
                for (int j = sourceStartCol, k = targetStartCol; j < sourceEndCol; j++, k++)
                {
                    target[targetStartRow + i, k] = source[i, j];
                }
            }
        }
    }
}
