//------------------------------------------------------------------------------
// <copyright file="DeepMatrix.cs" author="ameritusweb" date="5/14/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.Interprocess;

    /// <summary>
    /// A deep matrix class used for deep matrix operations.
    /// </summary>
    [Serializable]
    [JsonConverter(typeof(DeepMatrixJsonConverter))]
    public class DeepMatrix : IEnumerable<Matrix>, ICloneable
    {
        private Matrix[] matrices;

        /// <summary>
        /// Initializes a new instance of the <see cref="DeepMatrix"/> class.
        /// </summary>
        [JsonConstructor]
        public DeepMatrix()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DeepMatrix"/> class.
        /// </summary>
        /// <param name="depth">The depth.</param>
        /// <param name="rows">The rows.</param>
        /// <param name="cols">The cols.</param>
        public DeepMatrix(int depth, int rows, int cols)
        {
            this.matrices = new Matrix[depth];
            for (int i = 0; i < depth; ++i)
            {
                this.matrices[i] = new Matrix(rows, cols);
            }

            this.UniqueId = PseudoUniqueIDGenerator.Instance.GetNextID();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DeepMatrix"/> class.
        /// </summary>
        /// <param name="dimensions">The dimensions of the deep matrix.</param>
        public DeepMatrix(Dimension dimensions)
        {
            this.matrices = new Matrix[dimensions.Depth];
            for (int i = 0; i < dimensions.Depth; ++i)
            {
                this.matrices[i] = new Matrix(dimensions.Height, dimensions.Width);
            }

            this.UniqueId = PseudoUniqueIDGenerator.Instance.GetNextID();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DeepMatrix"/> class.
        /// </summary>
        /// <param name="matrices">The matrices to initialize with.</param>
        public DeepMatrix(Matrix[] matrices)
        {
            this.matrices = matrices;
            this.UniqueId = PseudoUniqueIDGenerator.Instance.GetNextID();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DeepMatrix"/> class.
        /// </summary>
        /// <param name="uniqueId">The unique ID.</param>
        /// <param name="matrixArrayValues">The matrices.</param>
        public DeepMatrix(int uniqueId, Matrix[] matrixArrayValues)
        {
            this.UniqueId = uniqueId;
            this.matrices = matrixArrayValues;
        }

        /// <summary>
        /// Gets the unique ID of the matrix.
        /// </summary>
        [JsonProperty]
        public int UniqueId { get; private set; }

        /// <summary>
        /// Gets the number of rows.
        /// </summary>
        public int Rows => this.matrices[0].Length;

        /// <summary>
        /// Gets the number of columns.
        /// </summary>
        public int Cols => this.matrices[0][0].Length;

        /// <summary>
        /// Gets the depth of the matrix.
        /// </summary>
        public int Depth => this.matrices.Length;

        /// <summary>
        /// Gets the count of the matrix.
        /// </summary>
        public int Count => this.matrices.Length;

        /// <summary>
        /// Gets the total count of rows for all matrices.
        /// </summary>
        public int TotalRows => this.matrices.Sum(m => m.Rows);

        /// <summary>
        /// Gets the dimension of the matrix.
        /// </summary>
        public Dimension Dimension => new Dimension(this.Depth, this.Rows, this.Cols);

        /// <summary>
        /// Gets the matrix values.
        /// </summary>
        [JsonProperty]
        internal Matrix[] MatrixArrayValues => this.matrices;

        /// <summary>
        /// Gets or sets the value at the specified row and column and depth.
        /// </summary>
        /// <param name="depth">The depth.</param>
        /// <param name="row">The row.</param>
        /// <param name="col">The column.</param>
        /// <returns>The value at the specified row and column and depth.</returns>
        public double this[int depth, int row, int col]
        {
            get { return this.matrices[depth][row][col]; }
            set { this.matrices[depth][row][col] = value; }
        }

        /// <summary>
        /// Gets or sets the matrix at the specified index.
        /// </summary>
        /// <param name="index">The matrix index.</param>
        /// <returns>The matrix.</returns>
        public Matrix this[int index]
        {
            get { return this.matrices[index]; }
            set { this.matrices[index] = value; }
        }

        /// <summary>
        /// Adds two deep matrices together.
        /// </summary>
        /// <param name="m1">The first deep matrix.</param>
        /// <param name="m2">The second deep matrix.</param>
        /// <returns>The resultant deep matrix.</returns>
        public static DeepMatrix operator +(DeepMatrix m1, DeepMatrix m2)
        {
            int depth = m1.Depth;
            int numRows = m1.Rows;
            int numCols = m1.Cols;
            DeepMatrix result = new DeepMatrix(depth, numRows, numCols);
            Parallel.For(0, depth, i =>
            {
                for (int j = 0; j < numRows; j++)
                {
                    for (int k = 0; k < numCols; ++k)
                    {
                        result[i, j, k] = m1[i, j, k] + m2[i, j, k];
                    }
                }
            });

            return result;
        }

        /// <summary>
        /// Multiplies a deep matrix with a scalar.
        /// </summary>
        /// <param name="m">The deep matrix to multiply.</param>
        /// <param name="scalar">The scalar value to multiply with.</param>
        /// <returns>The resultant deep matrix.</returns>
        public static DeepMatrix operator *(DeepMatrix m, double scalar)
        {
            int depth = m.Depth;
            int numRows = m.Rows;
            int numCols = m.Cols;

            DeepMatrix result = new DeepMatrix(depth, numRows, numCols);
            Parallel.For(0, depth, i =>
            {
                for (int j = 0; j < numRows; j++)
                {
                    for (int k = 0; k < numCols; ++k)
                    {
                        result[i, j, k] = m[i, j, k] * scalar;
                    }
                }
            });

            return result;
        }

        /// <summary>
        /// Initializes an array of deep matrices.
        /// </summary>
        /// <param name="size">The size.</param>
        /// <param name="depth">The depth.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <returns>An array of deep matrices.</returns>
        public static DeepMatrix[] InitializeArray(int size, int depth, int height, int width)
        {
            var instance = new DeepMatrix[size];
            for (int i = 0; i < size; ++i)
            {
                instance[i] = new DeepMatrix(depth, height, width);
            }

            return instance;
        }

        /// <summary>
        /// Initializes a double array of deep matrices.
        /// </summary>
        /// <param name="dim1">The first dimension.</param>
        /// <param name="dim2">The second dimension.</param>
        /// <param name="depth">The depth.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">THe width.</param>
        /// <returns>A double array of deep matrices.</returns>
        public static DeepMatrix[][] InitializeDoubleArray(int dim1, int dim2, int depth, int height, int width)
        {
            var instance = new DeepMatrix[dim1][];
            for (int i = 0; i < dim1; ++i)
            {
                instance[i] = new DeepMatrix[dim2];
                for (int j = 0; j < dim2; ++j)
                {
                    instance[i][j] = new DeepMatrix(depth, height, width);
                }
            }

            return instance;
        }

        /// <summary>
        /// Accumulates with the specified matrix array.
        /// </summary>
        /// <param name="matrixArray">The matrix array.</param>
        public void Accumulate(Matrix[] matrixArray)
        {
            for (int i = 0; i < this.Depth; ++i)
            {
                int rows = this[i].Rows;
                for (int j = 0; j < rows; ++j)
                {
                    int cols = this[i].Cols;
                    for (int k = 0; k < cols; ++k)
                    {
                        this[i][j][k] += matrixArray[i][j][k];
                    }
                }
            }
        }

        /// <summary>
        /// Computes the element-wise average of this deep matrix and another deep matrix.
        /// </summary>
        /// <param name="other">The other deep matrix.</param>
        /// <returns>A new deep matrix representing the element-wise average.</returns>
        public DeepMatrix Average(DeepMatrix other)
        {
            if (this.Depth != other.Depth || this.Rows != other.Rows || this.Cols != other.Cols)
            {
                throw new ArgumentException("Both matrices must have the same dimensions.");
            }

            DeepMatrix result = new DeepMatrix(this.Depth, this.Rows, this.Cols);

            Parallel.For(0, this.Depth, d =>
            {
                for (int i = 0; i < this.Rows; ++i)
                {
                    for (int j = 0; j < this.Cols; ++j)
                    {
                        result[d, i, j] = (this[d, i, j] + other[d, i, j]) / 2.0;
                    }
                }
            });

            return result;
        }

        /// <summary>
        /// Gets the deep matrix as an array of matrices.
        /// </summary>
        /// <returns>An array of matrices.</returns>
        public Matrix[] ToArray()
        {
            return this.matrices;
        }

        /// <summary>
        /// Replace the matrices with the specified matrices.
        /// </summary>
        /// <param name="matrices">The matrices to replace with.</param>
        public void Replace(Matrix[] matrices)
        {
            this.matrices = matrices;
        }

        /// <summary>
        /// Replaces the deep matrix with the specified rows.
        /// </summary>
        /// <param name="rows">The rows.</param>
        public void Replace(List<double[]> rows)
        {
            int skip = 0;
            for (int i = 0; i < this.Depth; ++i)
            {
                var mrows = this[i].Rows;
                this[i].Replace(rows.Skip(skip).Take(mrows).ToArray());
                skip += mrows;
            }
        }

        /// <summary>
        /// Clones the matrix.
        /// </summary>
        /// <returns>The cloned matrix.</returns>
        public object Clone()
        {
            return new DeepMatrix(this.matrices.Select(x => x.Clone()).OfType<Matrix>().ToArray());
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
                case InitializationType.Zeroes:
                    break;
                default:
                    throw new ArgumentException("Invalid initialization type.");
            }
        }

        /// <summary>
        /// Gets the enumerator for the matrix.
        /// </summary>
        /// <returns>The enumerator for the matrix.</returns>
        public IEnumerator<Matrix> GetEnumerator()
        {
            for (int i = 0; i < this.Depth; i++)
            {
                yield return this.matrices[i];
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

        private void InitializeHe()
        {
            var variance = 2.0 / this.Cols;

            Parallel.For(0, this.Depth, d =>
            {
                for (int i = 0; i < this.Rows; ++i)
                {
                    for (int j = 0; j < this.Cols; j++)
                    {
                        this[d, i, j] = Math.Sqrt(variance) * MatrixUtils.Random.NextDouble();
                    }
                }
            });
        }

        private void InitializeXavier()
        {
            Parallel.For(0, this.Depth, d =>
            {
                for (int i = 0; i < this.Rows; i++)
                {
                    for (int j = 0; j < this.Cols; j++)
                    {
                        this[d, i, j] = ((MatrixUtils.Random.NextDouble() * 2) - 1) * Math.Sqrt(6.0 / (this.Rows + this.Cols));
                    }
                }
            });
        }
    }
}
