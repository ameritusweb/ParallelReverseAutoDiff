//------------------------------------------------------------------------------
// <copyright file="MazeNode.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Amaze
{
    using ParallelReverseAutoDiff.RMAD;

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

        /// <summary>
        /// Calculates the indices.
        /// </summary>
        /// <param name="maze">The maze.</param>
        /// <returns>The matrix.</returns>
        public Matrix ToIndices(Maze maze)
        {
            List<double> indices = new List<double>();
            int index = 0;
            foreach (MazeDirectionType direction in Enum.GetValues(typeof(MazeDirectionType)))
            {
                if (this.AvailableDirections.Contains(direction))
                {
                    indices.Add(index++);
                }
                else
                {
                    indices.Add(1 + index++);
                }

                index++;
            }

            var quadrantIndicies = CubeSplitter.FindQuadrantIndices(new Point3d(this.PositionX, this.PositionY, this.PositionZ));
            indices.AddRange(quadrantIndicies.Select(x => (double)(index + x)));
            return new Matrix(indices.ToArray()).Transpose();
        }
    }
}
