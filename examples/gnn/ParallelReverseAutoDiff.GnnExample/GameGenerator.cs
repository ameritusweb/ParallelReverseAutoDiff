//------------------------------------------------------------------------------
// <copyright file="GameGenerator.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using Chess;

    /// <summary>
    /// Initializes a new instance of the <see cref="GameGenerator"/> class.
    /// </summary>
    public class GameGenerator
    {
        /// <summary>
        /// Generates for both chess engines.
        /// </summary>
        /// <returns>The PGN.</returns>
        public string GenerateBoth()
        {
            GameState gameState = new GameState();
            StockfishReader stockfishReader = new StockfishReader();
            RebelReader rebelReader = new RebelReader();
            while (true)
            {
                (string move, string ponder) = stockfishReader.ReadBestMove(gameState);
                if (!string.IsNullOrWhiteSpace(move))
                {
                    var res = this.MakeMove(gameState, move);
                    if (!res)
                    {
                        break;
                    }
                }

                if (gameState.IsGameOver() || gameState.Board.ExecutedMoves.Count > 149)
                {
                    break;
                }

                (string move2, string ponder2) = rebelReader.ReadBestMove(gameState);
                if (!string.IsNullOrWhiteSpace(move2))
                {
                    var res = this.MakeMove(gameState, move2);
                    if (!res)
                    {
                        break;
                    }
                }

                if (gameState.IsGameOver() || gameState.Board.ExecutedMoves.Count > 149)
                {
                    break;
                }
            }

            return gameState.Board.ToPgn();
        }

        /// <summary>
        /// Generates a chess game.
        /// </summary>
        /// <returns>The game in PGN format.</returns>
        public string Generate()
        {
            GameState gameState = new GameState();
            StockfishReader stockfishReader = new StockfishReader();
            while (true)
            {
                (string move, string ponder) = stockfishReader.ReadBestMove(gameState);
                if (!string.IsNullOrWhiteSpace(move))
                {
                    var res = this.MakeMove(gameState, move);
                    if (!res)
                    {
                        break;
                    }
                }

                if (!string.IsNullOrWhiteSpace(ponder))
                {
                    var res = this.MakeMove(gameState, ponder);
                    if (!res)
                    {
                        break;
                    }
                }

                if (gameState.IsGameOver() || gameState.Board.ExecutedMoves.Count > 99)
                {
                    break;
                }
            }

            return gameState.Board.ToPgn();
        }

        private bool MakeMove(GameState gameState, string move)
        {
            var position1 = move.Substring(0, 2);
            var position2 = move.Substring(2, 2);
            Position pos1 = new Position(position1);
            Position pos2 = new Position(position2);
            var piece = gameState.Board.GetPieceAt(pos1);
            Move move1 = new Move(pos1, pos2, piece);
            if (gameState.IsValidMove(move1))
            {
                gameState.Board.Move(move1);
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}
