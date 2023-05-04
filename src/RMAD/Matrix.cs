//------------------------------------------------------------------------------
// <copyright file="Matrix.cs" author="ameritusweb" date="5/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections;
    using System.Collections.Generic;
    using System.Threading.Tasks;

    /// <summary>
    /// A matrix class used for matrix operations.
    /// </summary>
    public class Matrix : IEnumerable<double[]>
    {
        private readonly double[][] matrix;

        /// <summary>
        /// Initializes a new instance of the <see cref="Matrix"/> class.
        /// </summary>
        /// <param name="rows">The number of rows.</param>
        /// <param name="cols">The number of cols.</param>
        public Matrix(int rows, int cols)
        {
            this.matrix = new double[rows][];
            for (int i = 0; i < rows; ++i)
            {
                this.matrix[i] = new double[cols];
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Matrix"/> class.
        /// </summary>
        /// <param name="matrix">The matrix to initialize with.</param>
        public Matrix(double[][] matrix)
        {
            this.matrix = matrix;
        }

        /// <summary>
        /// Gets the number of rows.
        /// </summary>
        public int Rows => this.matrix.Length;

        /// <summary>
        /// Gets the number of columns.
        /// </summary>
        public int Cols => this.matrix[0].Length;

        /// <summary>
        /// Gets the length of the matrix.
        /// </summary>
        public int Length => this.matrix.Length;

        /// <summary>
        /// Gets or sets the value at the specified row and column.
        /// </summary>
        /// <param name="row">The row.</param>
        /// <param name="col">The column.</param>
        /// <returns>The value at the specified row and column.</returns>
        public double this[int row, int col]
        {
            get { return this.matrix[row][col]; }
            set { this.matrix[row][col] = value; }
        }

        /// <summary>
        /// Gets or sets the row at the specified index.
        /// </summary>
        /// <param name="row">The row index.</param>
        /// <returns>The row.</returns>
        public double[] this[int row]
        {
            get { return this.matrix[row]; }
            set { this.matrix[row] = value; }
        }

        /// <summary>
        /// Adds two matrices together.
        /// </summary>
        /// <param name="m1">The first matrix.</param>
        /// <param name="m2">The second matrix.</param>
        /// <returns>The resultant matrix.</returns>
        public static Matrix operator +(Matrix m1, Matrix m2)
        {
            int numRows = m1.Rows;
            int numCols = m1.Cols;
            Matrix result = new Matrix(numRows, numCols);
            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    result[i, j] = m1[i, j] + m2[i, j];
                }
            });

            return result;
        }

        /// <summary>
        /// Multiplies two matrices together.
        /// </summary>
        /// <param name="m1">The first matrix.</param>
        /// <param name="m2">The second matrix.</param>
        /// <returns>The resultant matrix.</returns>
        public static Matrix operator *(Matrix m1, Matrix m2)
        {
            int numRows1 = m1.Rows;
            int numCols1 = m1.Cols;
            int numRows2 = m2.Rows;
            int numCols2 = m2.Cols;
            Matrix result = new Matrix(numRows1, numCols2);

            // Parallelize the outer loop
            Parallel.For(0, numRows1, i =>
            {
                for (int j = 0; j < numCols2; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < numCols1; k++)
                    {
                        sum += m1[i, k] * m2[k, j];
                    }

                    result[i, j] = sum;
                }
            });
            return result;
        }

        /// <summary>
        /// Gets the enumerator for the matrix.
        /// </summary>
        /// <returns>The enumerator for the matrix.</returns>
        public IEnumerator<double[]> GetEnumerator()
        {
            for (int i = 0; i < this.Rows; i++)
            {
                yield return this.matrix[i];
            }
        }

        /// <summary>
        /// Gets the enumerator for the matrix.
        /// </summary>
        /// <returns>The enumerator for the matrix.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }
    }
}
