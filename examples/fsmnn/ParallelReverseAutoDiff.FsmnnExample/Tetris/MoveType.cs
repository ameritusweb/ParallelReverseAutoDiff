//------------------------------------------------------------------------------
// <copyright file="MoveType.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Tetris
{
    /// <summary>
    /// The move type.
    /// </summary>
    public enum MoveType
    {
        /// <summary>
        /// No action.
        /// </summary>
        NoAction,

        /// <summary>
        /// Rotate clockwise.
        /// </summary>
        RotateClockwise,

        /// <summary>
        /// Rotate counter clockwise.
        /// </summary>
        RotateCounterClockwise,

        /// <summary>
        /// Move down.
        /// </summary>
        MoveDown,

        /// <summary>
        /// Move left.
        /// </summary>
        MoveLeft,

        /// <summary>
        /// Move right.
        /// </summary>
        MoveRight,
    }
}
