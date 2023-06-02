using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GnnExample
{
    public class StockfishReader
    {
        public void Read()
        {
            Process process = new Process();
            process.StartInfo.FileName = "stockfish.exe";
            process.StartInfo.UseShellExecute = false;
            process.StartInfo.RedirectStandardInput = true;
            process.StartInfo.RedirectStandardOutput = true;
            process.Start();

            StreamWriter streamWriter = process.StandardInput;
            StreamReader streamReader = process.StandardOutput;

            streamWriter.WriteLine("position startpos moves e2e4 e7e5 g1f3");
            streamWriter.WriteLine("go movetime 1000");

            string output = "";
            while (!output.Contains("bestmove"))
            {
                output += (char)streamReader.Read();
            }

            string bestMove = output.Substring(output.IndexOf("bestmove") + 9, 4);
            string promotionPiece = "";
            if (bestMove.Length == 5)
            {
                promotionPiece = bestMove[4].ToString();
            }

            streamWriter.Close();
            streamReader.Close();
            process.Close();

            Console.WriteLine(promotionPiece);
        }
    }
}
