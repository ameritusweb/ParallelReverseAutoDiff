//------------------------------------------------------------------------------
// <copyright file="TetrisPieceConfiguration.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Tetris
{
    /// <summary>
    /// A tetris piece configuration.
    /// </summary>
    public class TetrisPieceConfiguration
    {
        /// <summary>
        /// Gets or sets the shape.
        /// </summary>
        public TetrisShape Shape { get; set; }

        /// <summary>
        /// Gets or sets the flowers.
        /// </summary>
        public FlowerType[,] Flowers { get; set; } = new FlowerType[2, 2];

        /// <summary>
        /// Gets or sets the rotation.
        /// </summary>
        public int Rotation { get; set; } // 0, 90, 180, 270 degrees
    }
}
