//------------------------------------------------------------------------------
// <copyright file="MazeStore.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Amaze
{
    /// <summary>
    /// A maze store.
    /// </summary>
    public class MazeStore
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MazeStore"/> class.
        /// </summary>
        /// <param name="maze">The maze.</param>
        public MazeStore(Maze maze)
        {
            this.Size = maze.MazeNodes.GetLength(0);
            this.DirectionTypes = new bool[9 * this.Size * this.Size * this.Size];
            var enumValues = Enum.GetValues(typeof(MazeDirectionType));
            int index = 0;
            for (int x = 0; x < this.Size; x++)
            {
                for (int y = 0; y < this.Size; y++)
                {
                    for (int z = 0; z < this.Size; z++)
                    {
                        foreach (var value in enumValues)
                        {
                            if (maze.MazeNodes[x, y, z].AvailableDirections.Contains((MazeDirectionType)value))
                            {
                                this.DirectionTypes[index] = true;
                            }

                            index++;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Gets or sets the size.
        /// </summary>
        public int Size { get; set; }

        /// <summary>
        /// Gets or sets the direction types.
        /// </summary>
        public bool[] DirectionTypes { get; set; }

        /// <summary>
        /// Converts to a maze.
        /// </summary>
        /// <returns>The maze.</returns>
        public Maze ToMaze()
        {
            Maze maze = new Maze();
            maze.MazeNodes = new MazeNode[this.Size, this.Size, this.Size];
            int index = 0;

            // Cache the enum values and its length
            var enumValues = Enum.GetValues(typeof(MazeDirectionType));
            int enumLength = enumValues.Length;

            for (int x = 0; x < this.Size; x++)
            {
                for (int y = 0; y < this.Size; y++)
                {
                    for (int z = 0; z < this.Size; z++)
                    {
                        var availableDirections = new List<MazeDirectionType>(enumLength);  // Using List for dynamic size

                        foreach (MazeDirectionType value in enumValues)
                        {
                            if (this.DirectionTypes[index])
                            {
                                availableDirections.Add(value);
                            }

                            index++;
                        }

                        maze.MazeNodes[x, y, z] = new MazeNode() { PositionX = x, PositionY = y, PositionZ = z, AvailableDirections = availableDirections.ToArray() };
                    }
                }
            }

            return maze;
        }
    }
}
