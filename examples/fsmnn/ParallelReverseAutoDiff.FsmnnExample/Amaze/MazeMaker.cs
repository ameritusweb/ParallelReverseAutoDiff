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
        private readonly Random random;

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
            maze.Size = mazeSize;
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

            maze.MazePath = new MazePath { MazeNodes = new[] { maze.MazeNodes[0, 0, 0] } };

            return maze;
        }

        /// <summary>
        /// Change the path.
        /// </summary>
        /// <param name="maze">The maze.</param>
        /// <returns>The changed maze.</returns>
        public Maze ChangePath(Maze maze)
        {
            var first = maze.MazePath.MazeNodes[0].DeepClone();
            first.PositionX = 2;
            first.PositionY = 2;
            first.PositionZ = 2;
            Maze n = new Maze();
            n.MazePath = new MazePath { MazeNodes = new[] { first } };
            n.Size = maze.Size;
            n.MazeNodes = new MazeNode[maze.Size, maze.Size, maze.Size];
            for (int x = 0; x < maze.Size; x++)
            {
                for (int y = 0; y < maze.Size; y++)
                {
                    for (int z = 0; z < maze.Size; z++)
                    {
                        n.MazeNodes[x, y, z] = maze.MazeNodes[x, y, z].DeepClone();
                    }
                }
            }

            n.MazeNodes[0, 0, 0] = first;
            return n;
        }
    }
}
