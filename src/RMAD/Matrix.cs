//------------------------------------------------------------------------------
// <copyright file="Matrix.cs" author="ameritusweb" date="5/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Threading;
    using System.Threading.Tasks;
    using ParallelReverseAutoDiff.Interprocess;

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

            this.UniqueId = PseudoUniqueIDGenerator.Instance.GetNextID();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Matrix"/> class.
        /// </summary>
        /// <param name="matrix">The matrix to initialize with.</param>
        public Matrix(double[][] matrix)
        {
            this.matrix = matrix;
            this.UniqueId = PseudoUniqueIDGenerator.Instance.GetNextID();
        }

        /// <summary>
        /// Gets the unique ID of the matrix.
        /// </summary>
        public int UniqueId { get; private set; }

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

            if (numCols1 != numRows2)
            {
                throw new InvalidOperationException("Matrix dimensions must be compatible for multiplication.");
            }

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
        /// Deserialize from the buffer.
        /// </summary>
        /// <param name="buffer">The buffer.</param>
        /// <returns>A flat array.</returns>
        public static double[] DeserializeToFlatArray(ReadOnlySpan<byte> buffer)
        {
            // Skip the transpose flag, unique ID, rows, and columns
            int dataOffset = 1 + (3 * sizeof(int));

            int rows = BitConverter.ToInt32(buffer.Slice(1 + sizeof(int), sizeof(int)));
            int cols = BitConverter.ToInt32(buffer.Slice(1 + (2 * sizeof(int)), sizeof(int)));

            int elementsCount = rows * cols;
            double[] flatMatrix = new double[elementsCount];

            for (int i = 0; i < elementsCount; i++)
            {
                flatMatrix[i] = BitConverter.ToDouble(buffer.Slice((i * sizeof(double)) + dataOffset, sizeof(double)));
            }

            return flatMatrix;
        }

        /// <summary>
        /// Initializes the matrix with He or Xavier initialization.
        /// </summary>
        /// <param name="initializationType">The initialization type.</param>
        public void Initialize(InitializationType initializationType)
        {
            switch (initializationType)
            {
                case InitializationType.He:
                    this.InitializeHe();
                    break;
                case InitializationType.Xavier:
                    this.InitializeXavier();
                    break;
                default:
                    throw new ArgumentException("Invalid initialization type.");
            }
        }

        /// <summary>
        /// Serialize to the buffer.
        /// </summary>
        /// <param name="buffer">The buffer.</param>
        /// <param name="transpose">Whether to transpose the matrix.</param>
        public void Serialize(Span<byte> buffer, bool transpose)
        {
            // Write the transpose flag
            buffer[0] = (byte)(transpose ? 1 : 0);

            // Write the unique identifier
            BitConverter.TryWriteBytes(buffer.Slice(1, sizeof(int)), this.UniqueId);

            // Write the dimensions
            BitConverter.TryWriteBytes(buffer.Slice(1 + sizeof(int), sizeof(int)), this.Rows);
            BitConverter.TryWriteBytes(buffer.Slice(1 + (2 * sizeof(int)), sizeof(int)), this.Cols);

            // Write the matrix data as a flat array
            for (int i = 0; i < this.Rows; i++)
            {
                for (int j = 0; j < this.Cols; j++)
                {
                    int index = (i * this.Cols) + j;
                    BitConverter.TryWriteBytes(buffer.Slice((index * sizeof(double)) + (1 + (3 * sizeof(int))), sizeof(double)), this[i, j]);
                }
            }
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

        /// <summary>
        /// Overrides the equals operator.
        /// </summary>
        /// <param name="obj">The object to compare.</param>
        /// <returns>The comparison.</returns>
        public override bool Equals(object obj)
        {
            if (obj is Matrix other)
            {
                if (this.UniqueId == other.UniqueId)
                {
                    return true;
                }

                if (this.Rows != other.Rows || this.Cols != other.Cols)
                {
                    return false;
                }

                for (int i = 0; i < this.Rows; i++)
                {
                    for (int j = 0; j < this.Cols; j++)
                    {
                        if (this[i, j] != other[i, j])
                        {
                            return false;
                        }
                    }
                }

                return true;
            }

            return false;
        }

        /// <summary>
        /// Overrides the hash code.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode()
        {
            int hash = HashCode.Combine(this.UniqueId, this.Rows, this.Cols);

            for (int i = 0; i < this.Rows; i++)
            {
                for (int j = 0; j < this.Cols; j++)
                {
                    hash = HashCode.Combine(hash, this[i, j]);
                }
            }

            return hash;
        }

        private void InitializeHe()
        {
            Func<Random> randomFunc = () => new Random(Guid.NewGuid().GetHashCode());
            var localRandom = new ThreadLocal<Random>(randomFunc);
            var rand = localRandom == null || localRandom.Value == null ? randomFunc() : localRandom.Value;
            var variance = 2.0 / this.Cols;

            Parallel.For(0, this.Rows, i =>
            {
                for (int j = 0; j < this.Cols; j++)
                {
                    this[i, j] = Math.Sqrt(variance) * rand.NextDouble();
                }
            });
        }

        private void InitializeXavier()
        {
            Func<Random> randomFunc = () => new Random(Guid.NewGuid().GetHashCode());
            var localRandom = new ThreadLocal<Random>(randomFunc);
            var rand = localRandom == null || localRandom.Value == null ? randomFunc() : localRandom.Value;

            Parallel.For(0, this.Rows, i =>
            {
                for (int j = 0; j < this.Cols; j++)
                {
                    this[i, j] = ((rand.NextDouble() * 2) - 1) * Math.Sqrt(6.0 / (this.Rows + this.Cols));
                }
            });
        }
    }
}
