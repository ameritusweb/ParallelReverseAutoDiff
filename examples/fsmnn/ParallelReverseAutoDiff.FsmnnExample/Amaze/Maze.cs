//------------------------------------------------------------------------------
// <copyright file="Maze.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Amaze
{
    using ParallelReverseAutoDiff.GnnExample.Common;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Creates a maze.
    /// </summary>
    public class Maze
    {
        /// <summary>
        /// Gets or sets the maze nodes.
        /// </summary>
        public MazeNode[,,] MazeNodes { get; set; }

        /// <summary>
        /// Gets or sets the maze path.
        /// </summary>
        public MazePath MazePath { get; set; }

        /// <summary>
        /// Gets or sets the maze size.
        /// </summary>
        public int Size { get; set; }

        /// <summary>
        /// Gets or sets the maximum depth.
        /// </summary>
        public int MaxDepth { get; set; } = 4;

        /// <summary>
        /// Gets the node count.
        /// </summary>
        public int NodeCount
        {
            get
            {
                return this.Size * this.Size * this.Size;
            }
        }

        /// <summary>
        /// To the true label.
        /// </summary>
        /// <returns>The true label.</returns>
        public Matrix ToTrueLabel()
        {
            Matrix trueLabel = new Matrix(1, this.NodeCount);
            trueLabel[0, this.NodeCount - 1] = 1;
            return trueLabel;
        }

        /// <summary>
        /// To indices.
        /// </summary>
        /// <returns>The indices.</returns>
        public DeepMatrix ToIndices()
        {
            int numMatrices = this.MazePath.MazeNodes.Length;
            int numIndices = Enum.GetValues(typeof(MazeDirectionType)).Length + this.MaxDepth;
            DeepMatrix dm = new DeepMatrix(numMatrices, numIndices, 1);
            foreach (var (node, index) in this.MazePath.MazeNodes.WithIndex())
            {
                var indices = node.ToIndices(this);
                dm[index] = indices;
            }

            return dm;
        }
    }
}
