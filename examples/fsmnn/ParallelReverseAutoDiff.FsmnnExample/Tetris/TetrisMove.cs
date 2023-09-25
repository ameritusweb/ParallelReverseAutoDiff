//------------------------------------------------------------------------------
// <copyright file="TetrisMove.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Tetris
{
    /// <summary>
    /// A tetris move.
    /// </summary>
    public class TetrisMove
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TetrisMove"/> class.
        /// </summary>
        /// <param name="move">The move type.</param>
        public TetrisMove(MoveType move)
        {
            this.Move = move;
        }

        /// <summary>
        /// Gets or sets the move type.
        /// </summary>
        public MoveType Move { get; set; }

        /// <summary>
        /// Gets or sets the timestamp.
        /// </summary>
        public DateTimeOffset Timestamp { get; set; } = DateTimeOffset.UtcNow;
    }
}
