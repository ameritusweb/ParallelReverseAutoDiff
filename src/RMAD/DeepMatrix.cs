//------------------------------------------------------------------------------
// <copyright file="DeepMatrix.cs" author="ameritusweb" date="5/14/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections;
    using System.Collections.Generic;
    using ParallelReverseAutoDiff.Interprocess;

    /// <summary>
    /// A deep matrix class used for deep matrix operations.
    /// </summary>
    public class DeepMatrix : IEnumerable<Matrix>
    {
        private readonly Matrix[] matrices;

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
        /// <param name="matrices">The matrices to initialize with.</param>
        public DeepMatrix(Matrix[] matrices)
        {
            this.matrices = matrices;
            this.UniqueId = PseudoUniqueIDGenerator.Instance.GetNextID();
        }

        /// <summary>
        /// Gets the unique ID of the matrix.
        /// </summary>
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
    }
}
