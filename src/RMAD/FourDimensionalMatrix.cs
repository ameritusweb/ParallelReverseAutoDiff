//------------------------------------------------------------------------------
// <copyright file="FourDimensionalMatrix.cs" author="ameritusweb" date="5/14/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.Interprocess;

    /// <summary>
    /// A deep matrix class used for deep matrix operations.
    /// </summary>
    [Serializable]
    [JsonConverter(typeof(FourDimensionalMatrixJsonConverter))]
    public class FourDimensionalMatrix : IEnumerable<DeepMatrix>, IMatrix, ICloneable
    {
        private DeepMatrix[] matrices;

        /// <summary>
        /// Initializes a new instance of the <see cref="FourDimensionalMatrix"/> class.
        /// </summary>
        [JsonConstructor]
        public FourDimensionalMatrix()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="FourDimensionalMatrix"/> class.
        /// </summary>
        /// <param name="count">The count.</param>
        /// <param name="depth">The depth.</param>
        /// <param name="rows">The rows.</param>
        /// <param name="cols">The cols.</param>
        public FourDimensionalMatrix(int count, int depth, int rows, int cols)
        {
            this.matrices = new DeepMatrix[count];
            for (int i = 0; i < count; ++i)
            {
                this.matrices[i] = new DeepMatrix(depth, rows, cols);
            }

            this.UniqueId = PseudoUniqueIDGenerator.Instance.GetNextID();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="FourDimensionalMatrix"/> class.
        /// </summary>
        /// <param name="count">The count.</param>
        /// <param name="dimensions">The dimensions of the deep matrix.</param>
        public FourDimensionalMatrix(int count, Dimension dimensions)
        {
            this.matrices = new DeepMatrix[count];
            for (int i = 0; i < count; ++i)
            {
                this.matrices[i] = new DeepMatrix(dimensions);
            }

            this.UniqueId = PseudoUniqueIDGenerator.Instance.GetNextID();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="FourDimensionalMatrix"/> class.
        /// </summary>
        /// <param name="matrices">The deep matrices to initialize with.</param>
        public FourDimensionalMatrix(DeepMatrix[] matrices)
        {
            this.matrices = matrices;
            this.UniqueId = PseudoUniqueIDGenerator.Instance.GetNextID();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="FourDimensionalMatrix"/> class.
        /// </summary>
        /// <param name="uniqueId">The unique ID.</param>
        /// <param name="matrixArrayValues">The deep matrices.</param>
        public FourDimensionalMatrix(int uniqueId, DeepMatrix[] matrixArrayValues)
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
        public int Rows => this.matrices[0][0].Length;

        /// <summary>
        /// Gets the number of columns.
        /// </summary>
        public int Cols => this.matrices[0][0][0].Length;

        /// <summary>
        /// Gets the depth of the matrix.
        /// </summary>
        public int Depth => this.matrices[0].Depth;

        /// <summary>
        /// Gets the count of the matrix.
        /// </summary>
        public int Count => this.matrices.Length;

        /// <summary>
        /// Gets the dimension of the matrix.
        /// </summary>
        public Dimension Dimension => new Dimension(this.Depth, this.Rows, this.Cols);

        /// <summary>
        /// Gets the matrix values.
        /// </summary>
        [JsonProperty]
        internal DeepMatrix[] DeepMatrixArrayValues => this.matrices;

        /// <summary>
        /// Gets or sets the value at the specified row and column and depth.
        /// </summary>
        /// <param name="count">The count.</param>
        /// <param name="depth">The depth.</param>
        /// <param name="row">The row.</param>
        /// <param name="col">The column.</param>
        /// <returns>The value at the specified row and column and depth.</returns>
        public double this[int count, int depth, int row, int col]
        {
            get { return this.matrices[count][depth][row][col]; }
            set { this.matrices[count][depth][row][col] = value; }
        }

        /// <summary>
        /// Gets or sets the deep matrix at the specified index.
        /// </summary>
        /// <param name="index">The deep matrix index.</param>
        /// <returns>The deep matrix.</returns>
        public DeepMatrix this[int index]
        {
            get { return this.matrices[index]; }
            set { this.matrices[index] = value; }
        }

        /// <summary>
        /// Accumulate with the specified deep matrix array.
        /// </summary>
        /// <param name="deepMatrixArray">The deep matrix array.</param>
        public void Accumulate(DeepMatrix[] deepMatrixArray)
        {
            for (int i = 0; i < this.Count; ++i)
            {
                var depth = this[i].Depth;
                for (int j = 0; j < depth; ++j)
                {
                    var rows = this[i][j].Length;
                    for (int k = 0; k < rows; ++k)
                    {
                        var cols = this[i][j][k].Length;
                        for (int l = 0; l < cols; ++l)
                        {
                            this[i][j][k][l] += deepMatrixArray[i][j][k][l];
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Gets the deep matrix as an array of matrices.
        /// </summary>
        /// <returns>An array of matrices.</returns>
        public DeepMatrix[] ToArray()
        {
            return this.matrices;
        }

        /// <summary>
        /// Replace the matrices with the specified matrices.
        /// </summary>
        /// <param name="matrices">The matrices to replace with.</param>
        public void Replace(DeepMatrix[] matrices)
        {
            this.matrices = matrices;
        }

        /// <summary>
        /// Clones the matrix.
        /// </summary>
        /// <returns>The cloned matrix.</returns>
        public object Clone()
        {
            return new FourDimensionalMatrix(this.matrices.Select(x => x.Clone()).OfType<DeepMatrix>().ToArray());
        }

        /// <summary>
        /// Gets the enumerator for the matrix.
        /// </summary>
        /// <returns>The enumerator for the matrix.</returns>
        public IEnumerator<DeepMatrix> GetEnumerator()
        {
            for (int i = 0; i < this.Count; i++)
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
    }
}
