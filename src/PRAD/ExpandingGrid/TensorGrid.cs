//------------------------------------------------------------------------------
// <copyright file="TensorGrid.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.ExpandingGrid
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A tensor grid for an expanding grid GRU.
    /// </summary>
    public class TensorGrid
    {
        private const int GridSize = 11;
        private readonly Tensor[,] grid;
        private readonly int tensorSize;

        /// <summary>
        /// Initializes a new instance of the <see cref="TensorGrid"/> class.
        /// </summary>
        /// <param name="tensorSize">The size of the tensor.</param>
        public TensorGrid(int tensorSize)
        {
            this.tensorSize = tensorSize;
            this.grid = new Tensor[GridSize, GridSize];
        }

        /// <summary>
        /// Gets the tensor size.
        /// </summary>
        public int TensorSize => this.tensorSize;

        /// <summary>
        /// Extract the tensor based on the coordinates.
        /// </summary>
        /// <param name="x">The X coordinate.</param>
        /// <param name="y">The Y coordinate.</param>
        /// <returns>The tensor.</returns>
        /// <exception cref="IndexOutOfRangeException">Index out of range.</exception>
        public Tensor this[int x, int y]
        {
            get
            {
                if (x < 0 || x >= GridSize || y < 0 || y >= GridSize)
                {
                    throw new IndexOutOfRangeException("Index out of range for TensorGrid");
                }

                if (this.grid[x, y] == null)
                {
                    this.grid[x, y] = Tensor.XavierUniform(new int[] { this.tensorSize, this.tensorSize });
                }

                return this.grid[x, y];
            }
        }
    }
}
