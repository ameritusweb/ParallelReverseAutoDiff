//------------------------------------------------------------------------------
// <copyright file="TetrisGame.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Tetris
{
    /// <summary>
    /// The tetris game.
    /// </summary>
    public class TetrisGame
    {
        private static readonly Random Rand = new Random(Guid.NewGuid().GetHashCode());

        /// <summary>
        /// Initializes a new instance of the <see cref="TetrisGame"/> class.
        /// </summary>
        public TetrisGame()
        {
            this.Board = new TetrisBoard();
            this.GenerateUpcomingPieces();
            this.SpawnNewPiece();
        }

        /// <summary>
        /// Gets the board.
        /// </summary>
        public TetrisBoard Board { get; private set; }

        /// <summary>
        /// Gets or sets the current piece.
        /// </summary>
        public TetrisPiece CurrentPiece { get; set; }

        /// <summary>
        /// Gets the upcoming pieces.
        /// </summary>
        public List<TetrisPieceConfiguration> UpcomingPieces { get; private set; } = new List<TetrisPieceConfiguration>(5);

        private TetrisShapes TetrisShapes { get; } = new TetrisShapes();

        /// <summary>
        /// Spawn a new piece.
        /// </summary>
        public void SpawnNewPiece()
        {
            this.CurrentPiece = new TetrisPiece
            {
                Shape = this.UpcomingPieces[0].Shape,
                Rotation = this.UpcomingPieces[0].Rotation,
                Flowers = this.UpcomingPieces[0].Flowers,
                Position = (0, (TetrisBoard.Width / 2) - 1),
            };

            this.UpcomingPieces.RemoveAt(0);
            this.GenerateUpcomingPieces(1);  // Refill the upcoming pieces list
        }

        /// <summary>
        /// Is a valid move.
        /// </summary>
        /// <param name="piece">The piece.</param>
        /// <returns>True or false.</returns>
        public bool IsValidMove(TetrisPiece piece)
        {
            foreach (var block in this.GetAbsolutePositions(piece))
            {
                if (block.Row < 0 || block.Row >= TetrisBoard.Height || block.Col < 0 || block.Col >= TetrisBoard.Width)
                {
                    // Boundary check
                    return false;
                }

                if (this.Board.Cells[block.Row, block.Col].Flower != FlowerType.Empty)
                {
                    // Overlap check
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Move left.
        /// </summary>
        /// <returns>True or false.</returns>
        public bool MoveLeft()
        {
            TetrisPiece newPiece = this.CurrentPiece.Clone();
            newPiece.Position = (newPiece.Position.Col - 1, newPiece.Position.Col);
            if (this.IsValidMove(newPiece))
            {
                this.CurrentPiece = newPiece;
                return true;
            }

            return false;
        }

        /// <summary>
        /// Move right.
        /// </summary>
        /// <returns>True or false.</returns>
        public bool MoveRight()
        {
            TetrisPiece newPiece = this.CurrentPiece.Clone();
            newPiece.Position = (newPiece.Position.Col + 1, newPiece.Position.Col);
            if (this.IsValidMove(newPiece))
            {
                this.CurrentPiece = newPiece;
                return true;
            }

            return false;
        }

        /// <summary>
        /// Rotate the piece.
        /// </summary>
        /// <returns>True or false.</returns>
        public bool Rotate()
        {
            TetrisPiece newPiece = this.CurrentPiece.Clone();
            newPiece.Rotation = (newPiece.Rotation + 90) % 360;
            if (this.IsValidMove(newPiece))
            {
                this.CurrentPiece = newPiece;
                return true;
            }

            return false;
        }

        /// <summary>
        /// Perform a hard drop.
        /// </summary>
        /// <returns>True or false.</returns>
        public bool HardDrop()
        {
            TetrisPiece newPiece = this.CurrentPiece.Clone();
            while (true)
            {
                newPiece.Position = (newPiece.Position.Row + 1, newPiece.Position.Col);
                if (!this.IsValidMove(newPiece))
                {
                    // Move it back up one step since the last move caused a collision
                    newPiece.Position = (newPiece.Position.Row - 1, newPiece.Position.Col);
                    break;
                }
            }

            if (newPiece.Position.Row != this.CurrentPiece.Position.Row)
            {
                this.CurrentPiece = newPiece;
                return true;
            }

            return false;
        }

        /// <summary>
        /// Check for line clear.
        /// </summary>
        public void CheckForLineClear()
        {
            for (int row = TetrisBoard.Height - 1; row >= 0; row--)
            {
                bool isLineFull = true;

                for (int col = 0; col < TetrisBoard.Width; col++)
                {
                    if (this.Board.Cells[row, col].Flower == FlowerType.Empty)
                    {
                        isLineFull = false;
                        break;
                    }
                }

                if (isLineFull)
                {
                    // Clear the line and shift everything above downward
                    for (int r = row; r > 0; r--)
                    {
                        for (int col = 0; col < TetrisBoard.Width; col++)
                        {
                            this.Board.Cells[r, col].Flower = this.Board.Cells[r - 1, col].Flower;
                        }
                    }

                    // Clear the topmost row
                    for (int col = 0; col < TetrisBoard.Width; col++)
                    {
                        this.Board.Cells[0, col].Flower = FlowerType.Empty;
                    }

                    // Since we moved everything down, we need to re-check the current row
                    row++;
                }
            }

            if (this.CheckForFlowerBomb(out List<Cell> bombCells))
            {
                this.ActivateFlowerBombEffects(bombCells);
            }
        }

        /// <summary>
        /// Is game over.
        /// </summary>
        /// <returns>True or false.</returns>
        public bool IsGameOver()
        {
            TetrisPiece testPiece = new TetrisPiece
            {
                Shape = this.UpcomingPieces[0].Shape,
                Rotation = 0,
                Flowers = this.UpcomingPieces[0].Flowers,
                Position = (0, (TetrisBoard.Width / 2) - 1), // Starting position at the top-middle
            };
            return !this.IsValidMove(testPiece);
        }

        private static T RandomEnumValue<T>()
        {
            var values = Enum.GetValues(typeof(T));
            return (T)values.GetValue(Rand.Next(values.Length))!;
        }

        private bool CheckForFlowerBomb(out List<Cell> bombCells)
        {
            bombCells = new List<Cell>();

            for (int row = 0; row < TetrisBoard.Height; row++)
            {
                for (int col = 0; col < TetrisBoard.Width; col++)
                {
                    FlowerType flower = this.Board.Cells[row, col].Flower;

                    if (flower == FlowerType.Empty)
                    {
                        continue;
                    }

                    // Check horizontally
                    if (col <= TetrisBoard.Width - 4 &&
                        flower == this.Board.Cells[row, col + 1].Flower &&
                        flower == this.Board.Cells[row, col + 2].Flower &&
                        flower == this.Board.Cells[row, col + 3].Flower)
                    {
                        bombCells.Add(this.Board.Cells[row, col]);
                        bombCells.Add(this.Board.Cells[row, col + 1]);
                        bombCells.Add(this.Board.Cells[row, col + 2]);
                        bombCells.Add(this.Board.Cells[row, col + 3]);
                        return true;
                    }

                    // Check vertically
                    if (row <= TetrisBoard.Height - 4 &&
                        flower == this.Board.Cells[row + 1, col].Flower &&
                        flower == this.Board.Cells[row + 2, col].Flower &&
                        flower == this.Board.Cells[row + 3, col].Flower)
                    {
                        bombCells.Add(this.Board.Cells[row, col]);
                        bombCells.Add(this.Board.Cells[row + 1, col]);
                        bombCells.Add(this.Board.Cells[row + 2, col]);
                        bombCells.Add(this.Board.Cells[row + 3, col]);
                        return true;
                    }

                    // Check diagonal from top-left to bottom-right
                    if (row <= TetrisBoard.Height - 4 && col <= TetrisBoard.Width - 4 &&
                        flower == this.Board.Cells[row + 1, col + 1].Flower &&
                        flower == this.Board.Cells[row + 2, col + 2].Flower &&
                        flower == this.Board.Cells[row + 3, col + 3].Flower)
                    {
                        bombCells.Add(this.Board.Cells[row, col]);
                        bombCells.Add(this.Board.Cells[row + 1, col + 1]);
                        bombCells.Add(this.Board.Cells[row + 2, col + 2]);
                        bombCells.Add(this.Board.Cells[row + 3, col + 3]);
                        return true;
                    }

                    // Check diagonal from bottom-left to top-right
                    if (row >= 3 && col <= TetrisBoard.Width - 4 &&
                        flower == this.Board.Cells[row - 1, col + 1].Flower &&
                        flower == this.Board.Cells[row - 2, col + 2].Flower &&
                        flower == this.Board.Cells[row - 3, col + 3].Flower)
                    {
                        bombCells.Add(this.Board.Cells[row, col]);
                        bombCells.Add(this.Board.Cells[row - 1, col + 1]);
                        bombCells.Add(this.Board.Cells[row - 2, col + 2]);
                        bombCells.Add(this.Board.Cells[row - 3, col + 3]);
                        return true;
                    }
                }
            }

            return false;
        }

        private void ActivateFlowerBombEffects(List<Cell> bombCells)
        {
            // Check if the flower bomb is of type Rose
            if (bombCells[0].Flower == FlowerType.Rose)
            {
                foreach (var cell in bombCells)
                {
                    // Clear the entire row for each rose cell
                    for (int col = 0; col < TetrisBoard.Width; col++)
                    {
                        this.Board.Cells[cell.Position.Row, col].Flower = FlowerType.Empty;
                    }
                }
            }
            else
            {
                foreach (var cell in bombCells)
                {
                    // Destroy adjacent cells
                    this.DestroyAdjacentCells(cell.Position.Row, cell.Position.Col);
                }
            }
        }

        private void DestroyAdjacentCells(int row, int col)
        {
            // List of possible relative positions for adjacent cells
            int[] rowOffsets = { -1, 0, 1, 0 };
            int[] colOffsets = { 0, 1, 0, -1 };

            for (int i = 0; i < 4; i++)
            {
                int newRow = row + rowOffsets[i];
                int newCol = col + colOffsets[i];

                if (this.IsValidCell(newRow, newCol))
                {
                    this.Board.Cells[newRow, newCol].Flower = FlowerType.Empty;
                }
            }
        }

        private bool IsValidCell(int row, int col)
        {
            return row >= 0 && row < TetrisBoard.Height && col >= 0 && col < TetrisBoard.Width;
        }

        private void GenerateUpcomingPieces(int amount = 5)
        {
            // Populate UpcomingPieces list with random pieces and flowers
            for (int i = 0; i < amount; i++)
            {
                TetrisPieceConfiguration piece = new TetrisPieceConfiguration
                {
                    Shape = (TetrisShape)RandomEnumValue<TetrisShape>(),
                    Rotation = 0,
                    Flowers = this.GenerateRandomFlowers(),
                };
                this.UpcomingPieces.Add(piece);
            }
        }

        private FlowerType[,] GenerateRandomFlowers()
        {
            FlowerType[,] flowers = new FlowerType[2, 2];
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    flowers[i, j] = (FlowerType)RandomEnumValue<FlowerType>();
                }
            }

            return flowers;
        }

        private List<(int Row, int Col)> GetAbsolutePositions(TetrisPiece piece)
        {
            List<(int Row, int Col)> positions = new List<(int Row, int Col)>();

            int shapeIndex = (int)piece.Shape;
            int rotationIndex = piece.Rotation / 90;  // Convert degrees to index: 0, 1, 2, or 3

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    if (this.TetrisShapes[shapeIndex, rotationIndex, i, j])
                    {
                        positions.Add((piece.Position.Row + i, piece.Position.Col + j));
                    }
                }
            }

            return positions;
        }
    }
}
