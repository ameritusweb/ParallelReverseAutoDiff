//------------------------------------------------------------------------------
// <copyright file="EvaluationTable.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using Chess;

    /// <summary>
    /// The evaluation table.
    /// </summary>
    public class EvaluationTable
    {
        /// <summary>
        /// Gets or sets the evaluation squares.
        /// </summary>
        public EvaluationSquare[,] Squares { get; set; } = new EvaluationSquare[8, 8];

        /// <summary>
        /// Sets the table.
        /// </summary>
        /// <param name="state">The game state.</param>
        public void SetTable(GameState state)
        {
            // First phase: Initialize the EvaluationSquares and their pieces
            var positions = state.GetPiecesAndTheirPositions();
            foreach (var position in positions)
            {
                this.Squares[position.Position.Y, position.Position.X] = new EvaluationSquare()
                {
                    Piece = position.Piece,
                    Position = position.Position,
                };
            }

            // Second phase: Assign the neighbors
            foreach (var position in positions)
            {
                this.Squares[position.Position.Y, position.Position.X].SetNeighbors(
                    this.ToSquare(position.Position.Y + 1, position.Position.X),
                    this.ToSquare(position.Position.Y, position.Position.X - 1),
                    this.ToSquare(position.Position.Y - 1, position.Position.X),
                    this.ToSquare(position.Position.Y, position.Position.X + 1),
                    this.ToSquare(position.Position.Y + 1, position.Position.X - 1),
                    this.ToSquare(position.Position.Y + 1, position.Position.X + 1),
                    this.ToSquare(position.Position.Y - 1, position.Position.X - 1),
                    this.ToSquare(position.Position.Y - 1, position.Position.X + 1));
            }
        }

        /// <summary>
        /// Get the to square.
        /// </summary>
        /// <param name="rank">The rank.</param>
        /// <param name="file">The file.</param>
        /// <returns>The evaluation square.</returns>
        public EvaluationSquare? ToSquare(int rank, int file)
        {
            if (rank >= 0 && rank < 8 && file >= 0 && file < 8)
            {
                return this.Squares[rank, file];
            }
            else
            {
                return null;
            }
        }

        /// <summary>
        /// Pass the messages.
        /// </summary>
        public void PassMessages()
        {
            Parallel.ForEach(this.Squares.Cast<EvaluationSquare>(), square =>
            {
                square.PassMessages();
            });
        }

        /// <summary>
        /// Get the square control.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <param name="color">The color.</param>
        /// <returns>The square control.</returns>
        public int GetSquareControl(Position position, PieceColor color)
        {
            var square = this.Squares[position.Y, position.X];
            switch (color.Name)
            {
                case nameof(PieceColor.White):
                    return square.MaxWhiteEnumCount > 1 ? 1 : 0;
                case nameof(PieceColor.Black):
                    return square.MaxBlackEnumCount > 1 ? 1 : 0;
                default:
                    throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Gets the stacked score.
        /// </summary>
        /// <param name="color">The piece color.</param>
        /// <returns>The stacked score.</returns>
        public double GetStackedScore(PieceColor color)
        {
            var score = 0d;
            var scorePerStack = 0.25d;
            foreach (var square in this.Squares.Cast<EvaluationSquare>())
            {
                switch (color.Name)
                {
                    case nameof(PieceColor.White):
                        score += scorePerStack * Math.Max(0d, square.MaxWhiteEnumCount - 1d);
                        break;
                    case nameof(PieceColor.Black):
                        score += scorePerStack * Math.Max(0d, square.MaxBlackEnumCount - 1d);
                        break;
                    default:
                        throw new NotImplementedException();
                }
            }

            return score;
        }
    }
}
