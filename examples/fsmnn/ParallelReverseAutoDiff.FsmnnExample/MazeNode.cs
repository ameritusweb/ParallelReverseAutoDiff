//------------------------------------------------------------------------------
// <copyright file="MazeNode.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample
{
    /// <summary>
    /// A node in the maze.
    /// </summary>
    public class MazeNode
    {
        /// <summary>
        /// Gets or sets the available directions.
        /// </summary>
        public MazeDirectionType[] AvailableDirections { get; set; }

        /// <summary>
        /// Gets or sets the position X.
        /// </summary>
        public int PositionX { get; set; }

        /// <summary>
        /// Gets or sets the position Y.
        /// </summary>
        public int PositionY { get; set; }

        /// <summary>
        /// Gets or sets the position Z.
        /// </summary>
        public int PositionZ { get; set; }
    }
}
