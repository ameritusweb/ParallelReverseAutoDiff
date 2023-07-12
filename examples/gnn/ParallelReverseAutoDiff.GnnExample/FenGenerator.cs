//------------------------------------------------------------------------------
// <copyright file="FenGenerator.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using System.IO.Compression;
    using Chess;
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.GnnExample.Common;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths;

    /// <summary>
    /// A FEN generator.
    /// </summary>
    public class FenGenerator
    {
        private GameState gameState;

        private ChessBoardLoader loader;

        /// <summary>
        /// Initializes a new instance of the <see cref="FenGenerator"/> class.
        /// </summary>
        public FenGenerator()
        {
            this.gameState = new GameState();
            this.loader = new ChessBoardLoader();
        }

        /// <summary>
        /// Loads the data.
        /// </summary>
        public void LoadData()
        {
            int total = this.loader.GetTotal();
            for (int t = 0; t < total; ++t)
            {
                var moves = this.loader.LoadMoves(t);
                var name = this.loader.GetFileName(t).Replace(".pgn", string.Empty);
                this.gameState = new GameState();
                List<string> jsons = new List<string>();
                try
                {
                    foreach ((Move move, Move? nextmove) in moves.WithNext())
                    {
                        this.gameState.Board.Move(move);
                        if (nextmove != null)
                        {
                            var fen = this.gameState.Board.ToFen();
                            var legalmoves = this.gameState.GetAllMoves().Select(x => x.Piece.ToString() + x.OriginalPosition.ToString() + x.NewPosition.ToString()).Distinct().ToList();
                            var gJson = JsonConvert.SerializeObject((fen, legalmoves));
                            jsons.Add(gJson);
                        }
                    } // end foreach
                }
                catch (ChessGameEndedException)
                {
                    // ignore
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                    Console.WriteLine(ex.StackTrace);
                    Console.WriteLine($"Failed to process {name}");
                }

                this.SaveToZip(jsons, $"E:\\graphs\\{name}.zip");
            }
        }

        private void SaveToZip(List<string> jsons, string zipName)
        {
            List<byte[]> buffers = new List<byte[]>();
            foreach (var json in jsons)
            {
                buffers.Add(System.Text.Encoding.UTF8.GetBytes(json));
            }

            using (FileStream fileStream = new FileStream(zipName, FileMode.Create))
            {
                using (ZipArchive archive = new ZipArchive(fileStream, ZipArchiveMode.Create))
                {
                    foreach (var (buffer, index) in buffers.WithIndex())
                    {
                        Guid guid = Guid.NewGuid();
                        ZipArchiveEntry entry = archive.CreateEntry($"game{index}.json");
                        using (Stream entryStream = entry.Open())
                        {
                            using (GZipStream gzipStream = new GZipStream(entryStream, CompressionMode.Compress))
                            {
                                gzipStream.Write(buffer, 0, buffer.Length);
                            }
                        }
                    }
                }
            }
        }
    }
}
