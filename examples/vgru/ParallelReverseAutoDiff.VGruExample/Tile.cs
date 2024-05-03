//------------------------------------------------------------------------------
// <copyright file="Tile.cs" author="ameritusweb" date="5/2/2024">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A tile.
    /// </summary>
    public class Tile
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Tile"/> class.
        /// </summary>
        /// <param name="isPlaceholder">A placeholder.</param>
        public Tile(bool isPlaceholder = false)
        {
            this.IsPlaceholder = isPlaceholder;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Tile"/> class.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        public Tile(Matrix matrix)
        {
            this.Matrix = matrix;
        }

        /// <summary>
        /// Gets a value indicating whether this tile is a placeholder.
        /// </summary>
        public bool IsPlaceholder { get; }

        /// <summary>
        /// Gets or sets the matrix.
        /// </summary>
        public Matrix Matrix { get; set; }
    }
}
