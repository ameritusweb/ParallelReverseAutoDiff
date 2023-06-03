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
        private Random rand = new Random(Guid.NewGuid().GetHashCode());

        /// <summary>
        /// Generate for both chess engines and save.
        /// </summary>
        /// <param name="dir">The directory to save to.</param>
        /// <param name="iterations">The number of iterations.</param>
        public void GenerateBothAndSave(string dir, int iterations)
        {
            for (int i = 0; i < iterations; ++i)
            {
                var pgn = this.GenerateBoth();
                File.WriteAllText(dir + "\\stockfishwVrebelb" + Guid.NewGuid() + ".pgn", pgn);
            }
        }

        /// <summary>
        /// Generate for both chess engines and save switch.
        /// </summary>
        /// <param name="dir">The directory to save to.</param>
        /// <param name="iterations">The number of iterations.</param>
        public void GenerateBothAndSaveSwitch(string dir, int iterations)
        {
            for (int i = 0; i < iterations; ++i)
            {
                var pgn = this.GenerateBothOppositeStrict();
                File.WriteAllText(dir + "\\stockfishwVrebelbswitch" + Guid.NewGuid() + ".pgn", pgn);
            }
        }

        /// <summary>
        /// Generate for both chess engines and save opposite.
        /// </summary>
        /// <param name="dir">The directory to save to.</param>
        /// <param name="iterations">The number of iterations.</param>
        public void GenerateBothAndSaveOpposite(string dir, int iterations)
        {
            for (int i = 0; i < iterations; ++i)
            {
                var pgn = this.GenerateBothOpposite();
                File.WriteAllText(dir + "\\rebelwVstockfishb" + Guid.NewGuid() + ".pgn", pgn);
            }
        }

        /// <summary>
        /// Generates for both chess engines opposite strict.
        /// </summary>
        /// <returns>The PGN.</returns>
        public string GenerateBothOppositeStrict()
        {
            GameState gameState = new GameState();
            this.ApplyOpening(gameState);
            StockfishReader stockfishReader = new StockfishReader();
            RebelReader rebelReader = new RebelReader();
            while (true)
            {
                if (gameState.Board.ExecutedMoves.Count >= 70)
                {
                    (string move2, string ponder2) = rebelReader.ReadBestMove(gameState);
                    if (!string.IsNullOrWhiteSpace(move2))
                    {
                        var res = this.MakeMove(gameState, move2);
                        if (!res)
                        {
                            break;
                        }
                    }

                    if (gameState.IsGameOver() || gameState.Board.ExecutedMoves.Count > 179)
                    {
                        break;
                    }

                    (string move, string ponder) = stockfishReader.ReadBestMove(gameState);
                    if (!string.IsNullOrWhiteSpace(move))
                    {
                        var res = this.MakeMove(gameState, move);
                        if (!res)
                        {
                            break;
                        }
                    }

                    if (gameState.IsGameOver() || gameState.Board.ExecutedMoves.Count > 179)
                    {
                        break;
                    }
                }
                else
                {
                    (string move2, string ponder2) = stockfishReader.ReadBestMove(gameState);
                    if (!string.IsNullOrWhiteSpace(move2))
                    {
                        var res = this.MakeMove(gameState, move2);
                        if (!res)
                        {
                            break;
                        }
                    }

                    if (gameState.IsGameOver() || gameState.Board.ExecutedMoves.Count > 179)
                    {
                        break;
                    }

                    (string move, string ponder) = rebelReader.ReadBestMove(gameState);
                    if (!string.IsNullOrWhiteSpace(move))
                    {
                        var res = this.MakeMove(gameState, move);
                        if (!res)
                        {
                            break;
                        }
                    }

                    if (gameState.IsGameOver() || gameState.Board.ExecutedMoves.Count > 179)
                    {
                        break;
                    }
                }
            }

            return gameState.Board.ToPgn();
        }

        /// <summary>
        /// Generates for both chess engines opposite.
        /// </summary>
        /// <returns>The PGN.</returns>
        public string GenerateBothOpposite()
        {
            GameState gameState = new GameState();
            this.ApplyOpening(gameState);
            StockfishReader stockfishReader = new StockfishReader();
            RebelReader rebelReader = new RebelReader();
            while (true)
            {
                if (this.rand.NextDouble() <= 0.5d)
                {
                    (string move2, string ponder2) = rebelReader.ReadBestMove(gameState);
                    if (!string.IsNullOrWhiteSpace(move2))
                    {
                        var res = this.MakeMove(gameState, move2);
                        if (!res)
                        {
                            break;
                        }
                    }
                }
                else
                {
                    (string move2, string ponder2) = stockfishReader.ReadBestMove(gameState);
                    if (!string.IsNullOrWhiteSpace(move2))
                    {
                        var res = this.MakeMove(gameState, move2);
                        if (!res)
                        {
                            break;
                        }
                    }
                }

                if (gameState.IsGameOver() || gameState.Board.ExecutedMoves.Count > 179)
                {
                    break;
                }

                (string move, string ponder) = stockfishReader.ReadBestMove(gameState);
                if (!string.IsNullOrWhiteSpace(move))
                {
                    var res = this.MakeMove(gameState, move);
                    if (!res)
                    {
                        break;
                    }
                }

                if (gameState.IsGameOver() || gameState.Board.ExecutedMoves.Count > 179)
                {
                    break;
                }
            }

            return gameState.Board.ToPgn();
        }

        /// <summary>
        /// Generates for both chess engines.
        /// </summary>
        /// <returns>The PGN.</returns>
        public string GenerateBoth()
        {
            GameState gameState = new GameState();
            this.ApplyOpening(gameState);
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

                if (gameState.IsGameOver() || gameState.Board.ExecutedMoves.Count > 179)
                {
                    break;
                }

                if (this.rand.NextDouble() <= 0.5d)
                {
                    (string move2, string ponder2) = rebelReader.ReadBestMove(gameState);
                    if (!string.IsNullOrWhiteSpace(move2))
                    {
                        var res = this.MakeMove(gameState, move2);
                        if (!res)
                        {
                            break;
                        }
                    }
                }
                else
                {
                    (string move2, string ponder2) = stockfishReader.ReadBestMove(gameState);
                    if (!string.IsNullOrWhiteSpace(move2))
                    {
                        var res = this.MakeMove(gameState, move2);
                        if (!res)
                        {
                            break;
                        }
                    }
                }

                if (gameState.IsGameOver() || gameState.Board.ExecutedMoves.Count > 179)
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

        private void ApplyOpening(GameState gameState)
        {
            var openings = OpeningBook.GetOpenings();
            var keys = openings.Keys.ToList();
            var key = keys[this.rand.Next(keys.Count)];
            key = keys[54];
            var moves = openings[key];
            foreach (var move in moves)
            {
                bool valid = gameState.Board.Move(move);
                if (!valid)
                {
                }
            }
        }
    }
}
