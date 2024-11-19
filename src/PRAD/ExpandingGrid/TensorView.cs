//------------------------------------------------------------------------------
// <copyright file="TensorView.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.ExpandingGrid
{
    using System;

    /// <summary>
    /// Represents a view into a TensorGrid, providing access to a specific rectangular region of the grid.
    /// </summary>
    public class TensorView
    {
        private readonly TensorGrid baseGrid;
        private readonly int minX;
        private readonly int minY;
        private readonly int maxX;
        private readonly int maxY;

        /// <summary>
        /// Initializes a new instance of the <see cref="TensorView"/> class.
        /// </summary>
        /// <param name="grid">The base TensorGrid on which this view is created.</param>
        /// <param name="minX">The minimum X coordinate of the view in grid space.</param>
        /// <param name="minY">The minimum Y coordinate of the view in grid space.</param>
        /// <param name="maxX">The maximum X coordinate of the view in grid space.</param>
        /// <param name="maxY">The maximum Y coordinate of the view in grid space.</param>
        public TensorView(TensorGrid grid, int minX, int minY, int maxX, int maxY)
        {
            this.baseGrid = grid;
            this.minX = minX;
            this.minY = minY;
            this.maxX = maxX;
            this.maxY = maxY;
        }

        /// <summary>
        /// Gets the width of the view in data points.
        /// </summary>
        public int Width => (this.maxX - this.minX + 1) * this.baseGrid.TensorSize;

        /// <summary>
        /// Gets the height of the view in data points.
        /// </summary>
        public int Height => (this.maxY - this.minY + 1) * this.baseGrid.TensorSize;

        /// <summary>
        /// Gets or sets the value at the specified coordinates within the view.
        /// </summary>
        /// <param name="x">The X coordinate within the view.</param>
        /// <param name="y">The Y coordinate within the view.</param>
        /// <returns>The value at the specified coordinates.</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown when the coordinates are outside the view's bounds.</exception>
        public double this[int x, int y]
        {
            get
            {
                this.ValidateCoordinates(x, y);
                int gridX = (x / this.baseGrid.TensorSize) + this.minX;
                int gridY = (y / this.baseGrid.TensorSize) + this.minY;
                int localX = x % this.baseGrid.TensorSize;
                int localY = y % this.baseGrid.TensorSize;

                return this.baseGrid[gridX, gridY][localX, localY];
            }

            set
            {
                this.ValidateCoordinates(x, y);
                int gridX = (x / this.baseGrid.TensorSize) + this.minX;
                int gridY = (y / this.baseGrid.TensorSize) + this.minY;
                int localX = x % this.baseGrid.TensorSize;
                int localY = y % this.baseGrid.TensorSize;

                this.baseGrid[gridX, gridY][localX, localY] = value;
            }
        }

        /// <summary>
        /// Extracts all values from the tensor view into a one-dimensional array of doubles.
        /// The array is filled tensor by tensor, row by row within each tensor.
        /// </summary>
        /// <returns>A one-dimensional array containing all values from the view.</returns>
        public double[] Extract()
        {
            double[] result = new double[this.Width * this.Height];
            int destIndex = 0;

            for (int gridY = this.minY; gridY <= this.maxY; gridY++)
            {
                for (int tensorRow = 0; tensorRow < this.baseGrid.TensorSize; tensorRow++)
                {
                    for (int gridX = this.minX; gridX <= this.maxX; gridX++)
                    {
                        Tensor currentTensor = this.baseGrid[gridX, gridY];
                        double[] row = currentTensor.GetRow(tensorRow).Data;
                        Array.Copy(row, 0, result, destIndex, row.Length);
                        destIndex += row.Length;
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Sets the values in the tensor view from a one-dimensional array of doubles.
        /// The array is read tensor by tensor, row by row within each tensor.
        /// </summary>
        /// <param name="values">A one-dimensional array containing values to set in the view.</param>
        /// <exception cref="ArgumentException">Thrown when the input array size doesn't match the view size.</exception>
        public void ExtractReverse(double[] values)
        {
            if (values.Length != this.Width * this.Height)
            {
                throw new ArgumentException($"Input array size ({values.Length}) doesn't match the view size ({this.Width * this.Height}).");
            }

            int sourceIndex = 0;

            for (int gridY = this.minY; gridY <= this.maxY; gridY++)
            {
                for (int tensorRow = 0; tensorRow < this.baseGrid.TensorSize; tensorRow++)
                {
                    for (int gridX = this.minX; gridX <= this.maxX; gridX++)
                    {
                        Tensor currentTensor = this.baseGrid[gridX, gridY];
                        double[] row = new double[this.baseGrid.TensorSize];
                        Array.Copy(values, sourceIndex, row, 0, row.Length);
                        currentTensor.SetRow(tensorRow, row);
                        sourceIndex += row.Length;
                    }
                }
            }
        }

        /// <summary>
        /// Validates whether the given coordinates are within the view's bounds.
        /// </summary>
        /// <param name="x">The X coordinate to validate.</param>
        /// <param name="y">The Y coordinate to validate.</param>
        /// <exception cref="IndexOutOfRangeException">Thrown when the coordinates are outside the view's bounds.</exception>
        private void ValidateCoordinates(int x, int y)
        {
            if (x < 0 || x >= this.Width || y < 0 || y >= this.Height)
            {
                throw new IndexOutOfRangeException($"Coordinates ({x}, {y}) are outside the view's bounds of {this.Width}x{this.Height}.");
            }
        }
    }
}
