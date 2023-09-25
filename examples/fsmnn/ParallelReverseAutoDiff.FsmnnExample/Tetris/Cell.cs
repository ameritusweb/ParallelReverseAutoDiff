//------------------------------------------------------------------------------
// <copyright file="Cell.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Tetris
{
    /// <summary>
    /// The cell.
    /// </summary>
    public class Cell
    {
        /// <summary>
        /// Gets or sets the flower.
        /// </summary>
        public FlowerType Flower { get; set; }

        /// <summary>
        /// Gets or sets the shape.
        /// </summary>
        public TetrisShape Shape { get; set; }

        /// <summary>
        /// Gets or sets the position.
        /// </summary>
        public (int Row, int Col) Position { get; set; }
    }
}
