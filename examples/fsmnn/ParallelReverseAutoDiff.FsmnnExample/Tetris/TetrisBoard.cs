//------------------------------------------------------------------------------
// <copyright file="TetrisBoard.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Tetris
{
    /// <summary>
    /// The tetris board.
    /// </summary>
    public class TetrisBoard
    {
        /// <summary>
        /// The width of the board.
        /// </summary>
        public const int Width = 10;

        /// <summary>
        /// The height of the board.
        /// </summary>
        public const int Height = 20;

        /// <summary>
        /// Initializes a new instance of the <see cref="TetrisBoard"/> class.
        /// </summary>
        public TetrisBoard()
        {
            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                {
                    this.Cells[i, j] = new Cell { Flower = FlowerType.Empty, Shape = TetrisShape.None, Position = (i, j) };
                }
            }
        }

        /// <summary>
        /// Gets the cells of the board.
        /// </summary>
        public Cell[,] Cells { get; private set; } = new Cell[Height, Width];
    }
}
