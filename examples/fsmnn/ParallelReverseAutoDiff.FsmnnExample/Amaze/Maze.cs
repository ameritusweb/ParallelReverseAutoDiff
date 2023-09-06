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
        /// Gets or sets the number of quadrants.
        /// </summary>
        public int NumQuadrants { get; set; } = 8;

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
        /// Gets the number of indices.
        /// </summary>
        public int NumIndices
        {
            get
            {
                return Enum.GetValues(typeof(MazeDirectionType)).Length + this.MaxDepth;
            }
        }

        /// <summary>
        /// Gets the alphabet size.
        /// </summary>
        public int AlphabetSize
        {
            get
            {
                return (Enum.GetValues(typeof(MazeDirectionType)).Length * 2) + this.NumQuadrants;
            }
        }

        /// <summary>
        /// To the true label.
        /// </summary>
        /// <param name="size">The size.</param>
        /// <returns>The true label.</returns>
        public Matrix ToTrueLabel(int size)
        {
            Matrix trueLabel = new Matrix(1, size);
            trueLabel[0, size - 1] = 1;
            return trueLabel;
        }

        /// <summary>
        /// To indices.
        /// </summary>
        /// <returns>The indices.</returns>
        public DeepMatrix ToIndices()
        {
            int numMatrices = this.MazePath.MazeNodes.Length;
            int numIndices = this.NumIndices;
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
