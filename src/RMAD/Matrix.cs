﻿//------------------------------------------------------------------------------
// <copyright file="Matrix.cs" author="ameritusweb" date="5/4/2023">
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
    using ParallelReverseAutoDiff.PRAD;

    /// <summary>
    /// A matrix class used for matrix operations.
    /// </summary>
    [Serializable]
    [JsonConverter(typeof(MatrixJsonConverter))]
    public class Matrix : IEnumerable<double[]>, IMatrix, ICloneable
    {
        private static readonly double MaxSafeValue = Math.Log(double.MaxValue);  // The maximum value for which e^x results in double.MaxValue
        private double[][] matrix;
        private int[] shape;
        private int numDimensions;
        private long totalSize;

        /// <summary>
        /// Initializes a new instance of the <see cref="Matrix"/> class.
        /// </summary>
        [JsonConstructor]
        public Matrix()
        {
        }

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
            this.shape = new[] { rows, cols };
            this.numDimensions = 2;
            this.totalSize = rows * cols;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Matrix"/> class.
        /// </summary>
        /// <param name="batches">The number of batches.</param>
        /// <param name="rows">The number of rows.</param>
        /// <param name="cols">The number of cols.</param>
        public Matrix(int batches, int rows, int cols)
        {
            var rowSize = batches * rows;
            this.matrix = new double[rowSize][];
            for (int i = 0; i < rowSize; ++i)
            {
                this.matrix[i] = new double[cols];
            }

            this.UniqueId = PseudoUniqueIDGenerator.Instance.GetNextID();
            this.shape = new[] { batches, rows, cols };
            this.numDimensions = 3;
            this.totalSize = batches * rows * cols;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Matrix"/> class.
        /// </summary>
        /// <param name="matrix">The matrix to initialize with.</param>
        public Matrix(double[][] matrix)
        {
            this.matrix = matrix;
            this.UniqueId = PseudoUniqueIDGenerator.Instance.GetNextID();
            this.shape = new[] { matrix.Length, matrix[0].Length };
            this.numDimensions = 2;
            this.totalSize = this.shape.Aggregate(1, (a, b) => a * b);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Matrix"/> class.
        /// </summary>
        /// <param name="matrix">The matrix to initialize with.</param>
        /// <param name="shape">The shape of the matrix.</param>
        /// <param name="numDimensions">The number of dimensions.</param>
        public Matrix(double[][] matrix, int[] shape, int numDimensions)
        {
            this.matrix = matrix;
            this.UniqueId = PseudoUniqueIDGenerator.Instance.GetNextID();
            this.shape = shape;
            this.numDimensions = numDimensions;
            this.totalSize = this.shape.Aggregate(1, (a, b) => a * b);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Matrix"/> class with a 1xN matrix using a provided array.
        /// </summary>
        /// <param name="array">The array to initialize the matrix with.</param>
        public Matrix(double[] array)
        {
            this.matrix = new double[1][];
            this.matrix[0] = new double[array.Length];
            Array.Copy(array, this.matrix[0], array.Length);
            this.UniqueId = PseudoUniqueIDGenerator.Instance.GetNextID();
            this.shape = new[] { 1, array.Length };
            this.numDimensions = 2;
            this.totalSize = array.Length;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Matrix"/> class.
        /// </summary>
        /// <param name="uniqueId">The unique ID.</param>
        /// <param name="matrixValues">The matrix values.</param>
        public Matrix(int uniqueId, double[][] matrixValues)
        {
            this.UniqueId = uniqueId;
            this.matrix = matrixValues;
            this.shape = new[] { matrixValues.Length, matrixValues[0].Length };
            this.numDimensions = 2;
            this.totalSize = this.shape.Aggregate(1L, (a, b) => a * b);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Matrix"/> class.
        /// If the number of dimensions is greater than 2, the rows are stacked in the first dimension.
        /// </summary>
        /// <param name="uniqueId">The unique ID.</param>
        /// <param name="matrixValues">The matrix values.</param>
        /// <param name="shape">The shape of the matrix.</param>
        /// <param name="numDimensions">The number of dimensions.</param>
        public Matrix(int uniqueId, double[][] matrixValues, int[] shape, int numDimensions)
        {
            this.UniqueId = uniqueId;
            this.matrix = matrixValues;
            this.shape = shape;
            this.numDimensions = numDimensions;
            this.totalSize = this.shape.Aggregate(1L, (a, b) => a * b);
        }

        /// <summary>
        /// Gets the unique ID of the matrix.
        /// </summary>
        [JsonProperty]
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
        /// Gets the count of the matrix.
        /// </summary>
        public int Count => this.matrix.Length;

        /// <summary>
        /// Gets the shape of the matrix.
        /// </summary>
        [JsonProperty]
        public int[] Shape => this.shape;

        /// <summary>
        /// Gets the number of dimensions of the matrix.
        /// </summary>
        [JsonProperty]
        public int NumDimensions => this.numDimensions;

        /// <summary>
        /// Gets the matrix values.
        /// </summary>
        [JsonProperty]
        internal double[][] MatrixValues => this.matrix;

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
        /// Gets or sets the value at the specified indices.
        /// </summary>
        /// <param name="indices">The indices.</param>
        /// <returns>The value at the specified indices.</returns>
        public double this[int[] indices]
        {
            get
            {
                if (indices.Length != this.numDimensions)
                {
                    throw new ArgumentException("Number of indices must match the number of dimensions.");
                }

                return this.matrix[indices[0]][indices[1]];
            }

            set
            {
                if (indices.Length != this.numDimensions)
                {
                    throw new ArgumentException("Number of indices must match the number of dimensions.");
                }

                this.matrix[indices[0]][indices[1]] = value;
            }
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
        /// Subtracts two matrices.
        /// </summary>
        /// <param name="m1">The first matrix.</param>
        /// <param name="m2">The second matrix.</param>
        /// <returns>The resultant matrix.</returns>
        public static Matrix operator -(Matrix m1, Matrix m2)
        {
            int numRows = m1.Rows;
            int numCols = m1.Cols;
            Matrix result = new Matrix(numRows, numCols);
            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    result[i, j] = m1[i, j] - m2[i, j];
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
        /// Multiplies a matrix with a scalar.
        /// </summary>
        /// <param name="m">The matrix to multiply.</param>
        /// <param name="scalar">The scalar value to multiply with.</param>
        /// <returns>The resultant matrix.</returns>
        public static Matrix operator *(Matrix m, double scalar)
        {
            int numRows = m.Rows;
            int numCols = m.Cols;

            Matrix result = new Matrix(numRows, numCols);

            // Parallelize the outer loop
            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    result[i, j] = m[i, j] * scalar;
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
        /// Converts a matrix to a tensor.
        /// </summary>
        /// <returns>The tensor.</returns>
        public Tensor ToTensor()
        {
            double[] tensorData = new double[this.totalSize];

            for (int i = 0; i < this.Rows; i++)
            {
                Array.Copy(this.matrix[i], 0, tensorData, i * this.Cols, this.Cols);
            }

            return new Tensor(this.shape, tensorData);
        }

        /// <summary>
        /// Does the matrix contain any NaN or Infinity values.
        /// </summary>
        /// <returns>The result.</returns>
        public bool HasNaNOrInfinity()
        {
            for (int i = 0; i < this.Rows; i++)
            {
                for (int j = 0; j < this.Cols; j++)
                {
                    double value = this.matrix[i][j];
                    if (double.IsNaN(value) || double.IsInfinity(value))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        /// <summary>
        /// Sums each column across all rows to form a single row.
        /// </summary>
        /// <returns>An array of double values where each element is the sum of the corresponding column.</returns>
        public double[] ElementwiseSumRows()
        {
            if (this.Rows == 0)
            {
                throw new InvalidOperationException("Matrix has no rows.");
            }

            double[] columnSums = new double[this.Cols];
            for (int i = 0; i < this.Rows; i++)
            {
                for (int j = 0; j < this.Cols; j++)
                {
                    columnSums[j] += this.matrix[i][j];
                }
            }

            return columnSums;
        }

        /// <summary>
        /// Returns a new Matrix object containing the concatenated matrices.
        /// </summary>
        /// <param name="other">The other matrix.</param>
        /// <returns>New Matrix containing the concatenation.</returns>
        public Matrix ConcatenateColumns(Matrix other)
        {
            if (this.Rows != other.Rows)
            {
                throw new InvalidOperationException("Both matrices must have the same number of rows to concatenate columns.");
            }

            int newCols = this.Cols + other.Cols;
            Matrix result = new Matrix(this.Rows, newCols);

            // Parallelize the row copying
            Parallel.For(0, this.Rows, i =>
            {
                // Copy data from the current matrix
                Array.Copy(this.matrix[i], 0, result.matrix[i], 0, this.Cols);

                // Copy data from the other matrix
                Array.Copy(other.matrix[i], 0, result.matrix[i], this.Cols, other.Cols);
            });

            return result;
        }

        /// <summary>
        /// Returns a new Matrix object containing the ith column of the original Matrix.
        /// </summary>
        /// <param name="colIndex">The column index to slice.</param>
        /// <returns>New Matrix containing the column slice.</returns>
        public Matrix ColumnSlice(int colIndex)
        {
            if (colIndex >= this.Cols || colIndex < 0)
            {
                throw new IndexOutOfRangeException("Column index out of range.");
            }

            Matrix slice = new Matrix(this.Rows, 1);
            for (int i = 0; i < this.Rows; i++)
            {
                slice[i, 0] = this[i, colIndex];
            }

            return slice;
        }

        /// <summary>
        /// Sets the ith column of the Matrix to the values contained in the input Matrix.
        /// </summary>
        /// <param name="colIndex">The column index to set.</param>
        /// <param name="columnMatrix">The Matrix containing the values to set.</param>
        public void SetColumnSlice(int colIndex, Matrix columnMatrix)
        {
            if (colIndex >= this.Cols || colIndex < 0)
            {
                throw new IndexOutOfRangeException("Column index out of range.");
            }

            if (columnMatrix.Rows != this.Rows || columnMatrix.Cols != 1)
            {
                throw new ArgumentException("Input Matrix dimensions are not compatible.");
            }

            for (int i = 0; i < this.Rows; i++)
            {
                this[i, colIndex] = columnMatrix[i, 0];
            }
        }

        /// <summary>
        /// Performs element-wise power operation on the matrix.
        /// </summary>
        /// <param name="power">The power to raise each element to.</param>
        /// <returns>A new matrix with each element raised to the specified power.</returns>
        public Matrix ElementwisePower(double power)
        {
            int numRows = this.Rows;
            int numCols = this.Cols;
            Matrix result = new Matrix(numRows, numCols);

            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    result[i, j] = Math.Pow(this[i, j], power);
                }
            });

            return result;
        }

        /// <summary>
        /// Calculates the mean of all elements in the matrix.
        /// </summary>
        /// <returns>The mean of the matrix elements.</returns>
        public double Mean()
        {
            double sum = 0;
            int totalElements = this.Rows * this.Cols;

            foreach (var row in this.matrix)
            {
                sum += row.Sum();
            }

            return sum / totalElements;
        }

        /// <summary>
        /// Returns the transpose of this matrix.
        /// </summary>
        /// <returns>The transpose of this matrix.</returns>
        public Matrix Transpose()
        {
            int numRows = this.Cols;
            int numCols = this.Rows;
            Matrix result = new Matrix(numRows, numCols);

            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    result[i, j] = this[j, i];
                }
            });

            return result;
        }

        /// <summary>
        /// Accumulates with the specified double array.
        /// </summary>
        /// <param name="doubleArray">The double array.</param>
        public void Accumulate(double[][] doubleArray)
        {
            for (int i = 0; i < this.Rows; ++i)
            {
                int cols = this[i].Length;
                for (int j = 0; j < cols; ++j)
                {
                    this[i][j] += doubleArray[i][j];
                }
            }
        }

        /// <summary>
        /// Sums all the elements of the matrix.
        /// </summary>
        /// <returns>The sum of all the elements in the matrix.</returns>
        public double Sum()
        {
            int numRows = this.Rows;
            int numCols = this.Cols;
            double[] rowSums = new double[numRows];

            Parallel.For(0, numRows, i =>
            {
                double localRowSum = 0;
                for (int j = 0; j < numCols; ++j)
                {
                    localRowSum += this[i, j];
                }

                rowSums[i] = localRowSum;
            });

            return rowSums.Sum();
        }

        /// <summary>
        /// Find the frobenius norm of the matrix.
        /// </summary>
        /// <returns>The frobenius norm of the matrix.</returns>
        public double FrobeniusNorm()
        {
            int numRows = this.Rows;
            int numCols = this.Cols;
            double[] partitionSums = new double[numRows];

            Parallel.For(0, numRows, i =>
            {
                double localSum = 0;
                for (int j = 0; j < numCols; ++j)
                {
                    localSum += this[i, j] * this[i, j];
                }

                partitionSums[i] = localSum;
            });

            double totalSum = partitionSums.Sum();
            return Math.Sqrt(totalSum);
        }

        /// <summary>
        /// Performs an element-wise multiplication of two matrices.
        /// </summary>
        /// <param name="other">The other matrix.</param>
        /// <returns>The resultant matrix.</returns>
        public Matrix ElementwiseMultiply(Matrix other)
        {
            int numRows = other.Rows;
            int numCols = other.Cols;
            Matrix result = new Matrix(numRows, numCols);
            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    result[i, j] = this[i, j] * other[i, j];
                }
            });

            return result;
        }

        /// <summary>
        /// Applies the exponential function e^x element-wise to the matrix, with overflow checks.
        /// </summary>
        /// <returns>The resultant matrix with e^x applied to each element.</returns>
        public Matrix ExponentialElementwise()
        {
            int numRows = this.Rows;
            int numCols = this.Cols;
            Matrix result = new Matrix(numRows, numCols);

            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    if (this[i, j] > MaxSafeValue)
                    {
                        result[i, j] = double.MaxValue;
                    }
                    else
                    {
                        result[i, j] = Math.Exp(this[i, j]);
                    }
                }
            });

            return result;
        }

        /// <summary>
        /// Calculates the average of two matrices element-wise.
        /// </summary>
        /// <param name="other">The other matrix.</param>
        /// <returns>The resultant matrix.</returns>
        public Matrix Average(Matrix other)
        {
            if (this.Rows != other.Rows || this.Cols != other.Cols)
            {
                throw new ArgumentException("Matrices dimensions must be the same for averaging.");
            }

            Matrix result = new Matrix(this.Rows, this.Cols);

            Parallel.For(0, this.Rows, i =>
            {
                for (int j = 0; j < this.Cols; j++)
                {
                    result[i, j] = (this[i, j] + other[i, j]) / 2.0;
                }
            });

            return result;
        }

        /// <summary>
        /// Calculates the cosine similarity between two vectors.
        /// </summary>
        /// <param name="other">The other vector.</param>
        /// <returns>The cosine similarity.</returns>
        public double CosineSimilarity(Matrix other)
        {
            if (this.matrix.Length != other.matrix.Length || this.matrix[0].Length != 1 || other.matrix[0].Length != 1)
            {
                throw new ArgumentException("Both matrices must be vectors of the same size.");
            }

            double dotProduct = 0.0;
            double thisMagnitude = 0.0;
            double otherMagnitude = 0.0;

            for (int i = 0; i < this.matrix.Length; i++)
            {
                double thisValue = this.matrix[i][0];
                double otherValue = other.matrix[i][0];

                dotProduct += thisValue * otherValue;
                thisMagnitude += Math.Pow(thisValue, 2);
                otherMagnitude += Math.Pow(otherValue, 2);
            }

            if (thisMagnitude == 0.0 || otherMagnitude == 0.0)
            {
                throw new ArithmeticException("Cosine similarity is not defined if one or both of the vectors are zero-vectors.");
            }

            return dotProduct / (Math.Sqrt(thisMagnitude) * Math.Sqrt(otherMagnitude));
        }

        /// <summary>
        /// Calculate gradient wrt to the cosine similarity between two vectors.
        /// </summary>
        /// <param name="other">The other vector.</param>
        /// <param name="dLoss">The gradient of the loss.</param>
        /// <returns>The gradient wrt the cosine similarity.</returns>
        public Matrix GradientWRTCosineSimilarity(Matrix other, double dLoss)
        {
            if (this.matrix.Length != other.matrix.Length || this.matrix[0].Length != 1 || other.matrix[0].Length != 1)
            {
                throw new ArgumentException("Both matrices must be vectors of the same size.");
            }

            double dotProduct = 0.0;
            double thisMagnitude = 0.0;
            double otherMagnitude = 0.0;
            Matrix grad = new Matrix(this.matrix.Length, 1);

            for (int i = 0; i < this.matrix.Length; i++)
            {
                double thisValue = this.matrix[i][0];
                double otherValue = other.matrix[i][0];

                dotProduct += thisValue * otherValue;
                thisMagnitude += Math.Pow(thisValue, 2);
                otherMagnitude += Math.Pow(otherValue, 2);
            }

            for (int i = 0; i < this.matrix.Length; i++)
            {
                double thisValue = this.matrix[i][0];
                double otherValue = other.matrix[i][0];

                grad[i][0] = dLoss * ((otherValue / (Math.Sqrt(thisMagnitude) * Math.Sqrt(otherMagnitude))) - ((thisValue * dotProduct) / (Math.Pow(thisMagnitude, 1.5) * Math.Sqrt(otherMagnitude))));
            }

            return grad;
        }

        /// <summary>
        /// Retrieves the column at the specified index.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns>The column.</returns>
        public double[] Column(int index)
        {
            double[] column = new double[this.Rows];
            for (int i = 0; i < this.Rows; i++)
            {
                column[i] = this[i, index];
            }

            return column;
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
                case InitializationType.HeAdjacency:
                    this.InitializeHe(1.0d, 5.0d);
                    break;
                case InitializationType.Ones:
                    this.InitializeOnes();
                    break;
                case InitializationType.Zeroes:
                    break;
                default:
                    throw new ArgumentException("Invalid initialization type.");
            }
        }

        /// <summary>
        /// Initializes the matrix with He or Xavier initialization.
        /// </summary>
        /// <param name="initializationType">The initialization type.</param>
        /// <param name="scalingFactor">The scaling factor.</param>
        public void Initialize(InitializationType initializationType, double scalingFactor)
        {
            switch (initializationType)
            {
                case InitializationType.He:
                    this.InitializeHe(scalingFactor);
                    break;
                case InitializationType.Xavier:
                    this.InitializeXavier(scalingFactor);
                    break;
                case InitializationType.Ones:
                    this.InitializeOnes(scalingFactor);
                    break;
                case InitializationType.HeAdjacency:
                    this.InitializeHe(scalingFactor, 5.0d);
                    break;
                case InitializationType.Zeroes:
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
        /// Set the column at the specified index to the specified value.
        /// </summary>
        /// <param name="columnIndex">The column index.</param>
        /// <param name="value">The value to set.</param>
        public void SetColumn(int columnIndex, double value)
        {
            for (int i = 0; i < this.Rows; i++)
            {
                this[i, columnIndex] = value;
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
        /// Replace the matrix with the specified 2-D array.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        public void Replace(double[][] matrix)
        {
            this.matrix = matrix;
        }

        /// <summary>
        /// Replace the matrix with the specified 1-D array vertically.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        /// <returns>The matrix that was replaced.</returns>
        public Matrix ReplaceVertically(double[] matrix)
        {
            List<double[]> list = new List<double[]>();
            for (int i = 0; i < matrix.Length; ++i)
            {
                list.Add(new[] { matrix[i] });
            }

            this.matrix = list.ToArray();
            return this;
        }

        /// <summary>
        /// Retrieve the matrix as a 2-D array.
        /// </summary>
        /// <returns>The 2-D array.</returns>
        public double[][] ToArray()
        {
            return this.matrix;
        }

        /// <summary>
        /// Clones the matrix.
        /// </summary>
        /// <returns>The cloned matrix.</returns>
        public object Clone()
        {
            double[][] original = this.matrix;

            // Create new array of same length.
            double[][] deepCopy = new double[original.Length][];

            for (int i = 0; i < original.Length; i++)
            {
                // Allocate inner array and copy elements.
                deepCopy[i] = new double[original[i].Length];
                Array.Copy(original[i], deepCopy[i], original[i].Length);
            }

            return new Matrix(deepCopy);
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

        private void InitializeHe(double scalingFactor = 1.0, double shiftingFactor = 0.0)
        {
            var variance = 2.0 / this.Cols;

            Parallel.For(0, this.Rows, i =>
            {
                for (int j = 0; j < this.Cols; j++)
                {
                    this[i, j] = (Math.Sqrt(variance) * MatrixUtils.Random.NextDouble() * scalingFactor) + shiftingFactor;
                }
            });
        }

        private void InitializeXavier(double scalingFactor = 1.0)
        {
            Parallel.For(0, this.Rows, i =>
            {
                for (int j = 0; j < this.Cols; j++)
                {
                    this[i, j] = ((MatrixUtils.Random.NextDouble() * 2) - 1) * Math.Sqrt(6.0 / (this.Rows + this.Cols)) * scalingFactor;
                }
            });
        }

        private void InitializeOnes(double scalingFactor = 1.0)
        {
            Parallel.For(0, this.Rows, i =>
            {
                for (int j = 0; j < this.Cols; j++)
                {
                    this[i, j] = PradTools.One * scalingFactor;
                }
            });
        }
    }
}
