//------------------------------------------------------------------------------
// <copyright file="MazeDirectionType.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample
{
    /// <summary>
    /// The direction of the maze.
    /// </summary>
    public enum MazeDirectionType
    {
        /// <summary>
        /// Left.
        /// </summary>
        Left,

        /// <summary>
        /// Right.
        /// </summary>
        Right,

        /// <summary>
        /// Up.
        /// </summary>
        Up,

        /// <summary>
        /// Down.
        /// </summary>
        Down,

        /// <summary>
        /// Above.
        /// </summary>
        Above,

        /// <summary>
        /// Below.
        /// </summary>
        Below,

        /// <summary>
        /// Left or right.
        /// </summary>
        LeftOrRight,

        /// <summary>
        /// Up or down.
        /// </summary>
        UpOrDown,

        /// <summary>
        /// Left or right or up or down.
        /// </summary>
        LeftOrRightOrUpOrDown,
    }
}
