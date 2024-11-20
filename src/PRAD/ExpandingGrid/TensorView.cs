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
        private int minX;
        private int minY;
        private int maxX;
        private int maxY;

        /// <summary>
        /// Initializes a new instance of the <see cref="TensorView"/> class.
        /// </summary>
        /// <param name="grid">The base TensorGrid on which this view is created.</param>
        /// <param name="minX">The minimum X coordinate of the view in grid space.</param>
        /// <param name="minY">The minimum Y coordinate of the view in grid space.</param>
        /// <param name="maxX">The maximum X coordinate of the view in grid space.</param>
        /// <param name="maxY">The maximum Y coordinate of the view in grid space.</param>
        /// <param name="isPlaceholder">Is this a placeholder?.</param>
        public TensorView(TensorGrid grid, int minX, int minY, int maxX, int maxY, bool isPlaceholder = false)
        {
            this.baseGrid = grid;
            this.minX = minX;
            this.minY = minY;
            this.maxX = maxX;
            this.maxY = maxY;

            // when you instantiate it, that's when you instantiate the tensors.
            for (int i = minX; i <= maxX; ++i)
            {
                for (int j = minY; j <= maxY; ++j)
                {
                    if (this.baseGrid[i, j] == null)
                    {
                        if (isPlaceholder)
                        {
                            this.baseGrid.MakePlaceholder(grid.CenterIndex + i, grid.CenterIndex + j);
                        }
                        else
                        {
                            this.baseGrid.MakeXavierUniform(grid.CenterIndex + i, grid.CenterIndex + j);
                        }
                    }
                }
            }
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
        /// Expand in a certain direction and keeps rectangular using placeholders.
        /// </summary>
        /// <param name="expandFromX">The index to expand from X.</param>
        /// <param name="expandFromY">The index to expand from Y.</param>
        /// <param name="direction">The direction to expand in.</param>
        public void Expand(int expandFromX, int expandFromY, ExpansionDirection direction)
        {
            int destinationX = expandFromX;
            int destinationY = expandFromY;
            switch (direction)
            {
                case ExpansionDirection.Up:
                    destinationY--;
                    break;
                case ExpansionDirection.Down:
                    destinationY++;
                    break;
                case ExpansionDirection.Right:
                    destinationX++;
                    break;
                case ExpansionDirection.Left:
                    destinationX--;
                    break;
            }

            if (this.baseGrid[this.baseGrid.CenterIndex + destinationX, this.baseGrid.CenterIndex + destinationY] == null)
            {
                this.baseGrid.MakeXavierUniform(this.baseGrid.CenterIndex + destinationX, this.baseGrid.CenterIndex + destinationY);
            }

            if (destinationY > this.maxY)
            {
                this.maxY = destinationY;
                for (int i = this.minX; i <= this.maxX; ++i)
                {
                    if (this.baseGrid[this.baseGrid.CenterIndex + i, this.baseGrid.CenterIndex + destinationY] == null)
                    {
                        this.baseGrid.MakePlaceholder(this.baseGrid.CenterIndex + i, this.baseGrid.CenterIndex + destinationY);
                    }
                }
            }
            else if (destinationY < this.minY)
            {
                this.minY = destinationY;
                for (int i = this.minX; i <= this.maxX; ++i)
                {
                    if (this.baseGrid[this.baseGrid.CenterIndex + i, this.baseGrid.CenterIndex + destinationY] == null)
                    {
                        this.baseGrid.MakePlaceholder(this.baseGrid.CenterIndex + i, this.baseGrid.CenterIndex + destinationY);
                    }
                }
            }
            else if (destinationX > this.maxX)
            {
                this.maxX = destinationX;
                for (int i = this.minY; i <= this.maxY; ++i)
                {
                    if (this.baseGrid[this.baseGrid.CenterIndex + destinationX, this.baseGrid.CenterIndex + i] == null)
                    {
                        this.baseGrid.MakePlaceholder(this.baseGrid.CenterIndex + destinationX, this.baseGrid.CenterIndex + i);
                    }
                }
            }
            else if (destinationX < this.minX)
            {
                this.minX = destinationX;
                for (int i = this.minY; i <= this.maxY; ++i)
                {
                    if (this.baseGrid[this.baseGrid.CenterIndex + destinationX, this.baseGrid.CenterIndex + i] == null)
                    {
                        this.baseGrid.MakePlaceholder(this.baseGrid.CenterIndex + destinationX, this.baseGrid.CenterIndex + i);
                    }
                }
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
                        Tensor currentTensor = this.baseGrid[this.baseGrid.CenterIndex + gridX, this.baseGrid.CenterIndex + gridY];
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
                        Tensor currentTensor = this.baseGrid[this.baseGrid.CenterIndex + gridX, this.baseGrid.CenterIndex + gridY];
                        double[] row = new double[this.baseGrid.TensorSize];
                        Array.Copy(values, sourceIndex, row, 0, row.Length);
                        currentTensor.SetRow(tensorRow, row);
                        sourceIndex += row.Length;
                    }
                }
            }
        }
    }
}
