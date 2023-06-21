//------------------------------------------------------------------------------
// <copyright file="LeelaReader.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using System.Diagnostics;

    /// <summary>
    /// Reads from Leela.
    /// </summary>
    public class LeelaReader
    {
        private Random rand = new Random(Guid.NewGuid().GetHashCode());

        /// <summary>
        /// Reads the best move from Stockfish.
        /// </summary>
        /// <param name="gameState">The game state.</param>
        /// <returns>The move and the next move.</returns>
        public (string Move, string Ponder) ReadBestMove(GameState gameState)
        {
            Process process = new Process();
            process.StartInfo.WorkingDirectory = System.IO.Directory.GetCurrentDirectory() + "\\Leela";
            process.StartInfo.FileName = "cmd.exe";
            process.StartInfo.WindowStyle = ProcessWindowStyle.Hidden;
            process.StartInfo.Arguments = "/C lc0.exe";
            process.StartInfo.UseShellExecute = false;
            process.StartInfo.RedirectStandardInput = true;
            process.StartInfo.RedirectStandardOutput = true;
            process.Start();

            StreamWriter streamWriter = process.StandardInput;
            StreamReader streamReader = process.StandardOutput;

            streamWriter.WriteLine("uci");
            streamWriter.WriteLine("ucinewgame");
            streamWriter.WriteLine("setoption Ponder type true");
            streamWriter.WriteLine("isready");
            streamWriter.WriteLine("position fen " + gameState.Board.ToFen());
            streamWriter.WriteLine("go movetime " + this.rand.Next(2000, 5000));

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
