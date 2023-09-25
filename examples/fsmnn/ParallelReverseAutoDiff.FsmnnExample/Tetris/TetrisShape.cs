//------------------------------------------------------------------------------
// <copyright file="TetrisShape.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Tetris
{
    /// <summary>
    /// A tetris shape.
    /// </summary>
    public enum TetrisShape
    {
        /// <summary>
        /// No shape.
        /// </summary>
        None,

        /// <summary>
        /// A line piece.
        /// </summary>
        I,

        /// <summary>
        /// A square.
        /// </summary>
        O,

        /// <summary>
        /// A T.
        /// </summary>
        T,

        /// <summary>
        /// An S.
        /// </summary>
        S,

        /// <summary>
        /// A Z.
        /// </summary>
        Z,

        /// <summary>
        /// A J.
        /// </summary>
        J,

        /// <summary>
        /// An L.
        /// </summary>
        L,
    }
}
