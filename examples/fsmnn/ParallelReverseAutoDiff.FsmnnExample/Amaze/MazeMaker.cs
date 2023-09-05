//------------------------------------------------------------------------------
// <copyright file="MazeMaker.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Amaze
{
    /// <summary>
    /// Creates a maze.
    /// </summary>
    public class MazeMaker
    {
        private Random random;

        /// <summary>
        /// Initializes a new instance of the <see cref="MazeMaker"/> class.
        /// </summary>
        public MazeMaker()
        {
            this.random = new Random(Guid.NewGuid().GetHashCode());
        }

        /// <summary>
        /// Creates a maze.
        /// </summary>
        /// <param name="mazeSize">The maze size.</param>
        /// <returns>The maze.</returns>
        public Maze CreateMaze(int mazeSize)
        {
            var maze = new Maze();
            maze.MazeNodes = new MazeNode[mazeSize, mazeSize, mazeSize];

            // Cache the enum values
            var enumValues = Enum.GetValues(typeof(MazeDirectionType));

            for (int x = 0; x < mazeSize; x++)
            {
                for (int y = 0; y < mazeSize; y++)
                {
                    for (int z = 0; z < mazeSize; z++)
                    {
                        List<MazeDirectionType> types = new List<MazeDirectionType>();
                        foreach (MazeDirectionType value in enumValues)
                        {
                            if (this.random.Next(0, 2) == 0)
                            {
                                types.Add(value);
                            }
                        }

                        maze.MazeNodes[x, y, z] = new MazeNode { PositionX = x, PositionY = y, PositionZ = z, AvailableDirections = types.ToArray() };
                    }
                }
            }

            return maze;
        }
    }
}
