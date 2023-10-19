//------------------------------------------------------------------------------
// <copyright file="StockfishReader.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using System.Diagnostics;

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
    }
}
