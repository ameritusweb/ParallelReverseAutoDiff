//------------------------------------------------------------------------------
// <copyright file="StockfishReader.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using System;
    using System.Diagnostics;
    using System.IO;
    using System.Text;
    using System.Text.RegularExpressions;
    using Chess;

    /// <summary>
    /// Reads from Stockfish.
    /// </summary>
    public class StockfishReader
    {
        private readonly Random rand = new Random(Guid.NewGuid().GetHashCode());

        /// <summary>
        /// Reads the best move from Stockfish.
        /// </summary>
        /// <param name="gameState">The game state.</param>
        /// <returns>The move and the next move.</returns>
        public (string Move, string Ponder) ReadBestMove(GameState gameState)
        {
            Process process = new Process();
            process.StartInfo.WorkingDirectory = System.IO.Directory.GetCurrentDirectory() + "\\Stockfish15";
            process.StartInfo.FileName = "cmd.exe";
            process.StartInfo.WindowStyle = ProcessWindowStyle.Hidden;
            process.StartInfo.Arguments = "/C stockfish-windows-2022-x86-64-avx2.exe";
            process.StartInfo.UseShellExecute = false;
            process.StartInfo.RedirectStandardInput = true;
            process.StartInfo.RedirectStandardOutput = true;
            process.Start();

            StreamWriter streamWriter = process.StandardInput;
            StreamReader streamReader = process.StandardOutput;

            streamWriter.WriteLine("uci");
            streamWriter.WriteLine("ucinewgame");
            streamWriter.WriteLine("setoption name Threads value 8");
            streamWriter.WriteLine("setoption name Hash value 128");
            streamWriter.WriteLine("setoption name Skill Level value 20");
            streamWriter.WriteLine("position fen " + gameState.Board.ToFen());
            streamWriter.WriteLine("go movetime " + this.rand.Next(100, 2000));

            string output = string.Empty;
            while (true)
            {
                string? line = streamReader.ReadLine();
                if (line == null)
                {
                    break;
                }

                output += line;
                if (line.Contains("bestmove"))
                {
                    break;
                }
            }

            output += "  ";
            string bestMove = output.Substring(output.IndexOf("bestmove") + 9, 5).Trim();
            string ponder = string.Empty;

            int ponderIndex = output.IndexOf("ponder");
            if (ponderIndex != -1)
            {
                ponder = output.Substring(ponderIndex + 7, 5).Trim();
            }

            streamWriter.Close();
            streamReader.Close();
            process.Close();

            return (bestMove, ponder);
        }

        /// <summary>
        /// Reads the best move from Stockfish.
        /// </summary>
        /// <param name="gameState">The game state.</param>
        /// <param name="depth">The depth.</param>
        /// <returns>The best move score.</returns>
        public async Task<int> ReadBestMoveScoreAsync(GameState gameState, int depth = 20)
        {
            if (gameState.Board.IsEndGameCheckmate)
            {
                return gameState.Board.Turn == PieceColor.White ? -1000000000 : 1000000000;
            }
            else if (gameState.Board.EndGame != null)
            {
                return gameState.Board.Turn == PieceColor.White ? -1 : 1;
            }

            Process? process = null;
            StreamWriter? streamWriter = null;
            StreamReader? streamReader = null;

            try
            {
                process = new Process();
                process.StartInfo.WorkingDirectory = System.IO.Directory.GetCurrentDirectory() + "\\Stockfish15";
                process.StartInfo.FileName = "cmd.exe";
                process.StartInfo.WindowStyle = ProcessWindowStyle.Hidden;
                process.StartInfo.Arguments = "/C stockfish-windows-2022-x86-64-avx2.exe";
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.RedirectStandardInput = true;
                process.StartInfo.RedirectStandardOutput = true;
                process.Start();

                streamWriter = process.StandardInput;
                streamReader = process.StandardOutput;

                streamWriter.WriteLine("uci");
                streamWriter.WriteLine("ucinewgame");
                streamWriter.WriteLine("setoption name Threads value 8");
                streamWriter.WriteLine("setoption name Hash value 128");
                streamWriter.WriteLine("setoption name Skill Level value 20");
                streamWriter.WriteLine("isready");

                while (streamReader.ReadLine() != "readyok")
                {
                    /* Wait for readiness */
                }

                CancellationTokenSource cts = new CancellationTokenSource();
                cts.CancelAfter(TimeSpan.FromSeconds(500));

                StringBuilder sb = new StringBuilder("position fen " + gameState.Board.ToFen());
                await streamWriter.WriteLineAsync(sb, cts.Token);
                await streamWriter.WriteLineAsync("go depth " + depth);

                string? output;
                int score = 0;
                while ((output = await streamReader.ReadLineAsync()) != null)
                {
                    if (output.StartsWith("info depth " + depth.ToString()))
                    {
                        var match = Regex.Match(output, @"score cp (\S+)");
                        var mateMatch = Regex.Match(output, @"score mate (\S+)");

                        if (match.Success)
                        {
                            score = int.Parse(match.Groups[1].Value);
                            break;
                        }
                        else if (mateMatch.Success)
                        {
                            // A large constant is used for mates. Positive if it's good for white, negative if it's good for black
                            int mateIn = int.Parse(mateMatch.Groups[1].Value);

                            // A high constant divided by the number of moves until mate. The score is higher if fewer moves are left until checkmate.
                            // Positive if it's good for the engine (it can force a mate), negative if it's bad (the opponent can force a mate).
                            score = mateIn > 0 ? 100000000 / mateIn : -100000000 / Math.Abs(mateIn);
                            break;
                        }
                    }
                }

                return score;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            finally
            {
                streamWriter?.Close();
                streamReader?.Close();
                process?.Close();
            }

            return -1;
        }
    }
}
