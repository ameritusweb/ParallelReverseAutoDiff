//------------------------------------------------------------------------------
// <copyright file="TetrisPiece.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Tetris
{
    /// <summary>
    /// A tetris piece.
    /// </summary>
    public class TetrisPiece : TetrisPieceConfiguration
    {
        /// <summary>
        /// Gets or sets the position.
        /// </summary>
        public (int Row, int Col) Position { get; set; }

        /// <summary>
        /// Clone the tetris piece.
        /// </summary>
        /// <returns>The cloned piece.</returns>
        public TetrisPiece Clone()
        {
            return new TetrisPiece
            {
                Shape = this.Shape,
                Flowers = this.Flowers,
                Rotation = this.Rotation,
                Position = this.Position,
            };
        }
    }
}
