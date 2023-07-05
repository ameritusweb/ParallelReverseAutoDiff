//------------------------------------------------------------------------------
// <copyright file="TrainingSetGenerator.cs" author="ameritusweb" date="5/21/2023">
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
    /// A training set generator.
    /// </summary>
    public class TrainingSetGenerator
    {
        private GameState gameState;

        private ChessBoardLoader loader;

        private Dictionary<GamePhase, Dictionary<string, int>> edgeFrequenciesByPhase;

        private Dictionary<GamePhase, Dictionary<string, int>> moveFrequenciesByPhase;

        private Dictionary<GamePhase, Dictionary<string, int>> actualMoveFrequenciesByPhase;

        private Dictionary<string, int> artifacts;

        /// <summary>
        /// Initializes a new instance of the <see cref="TrainingSetGenerator"/> class.
        /// </summary>
        public TrainingSetGenerator()
        {
            this.gameState = new GameState();
            this.loader = new ChessBoardLoader();
            this.edgeFrequenciesByPhase = new Dictionary<GamePhase, Dictionary<string, int>>();
            this.moveFrequenciesByPhase = new Dictionary<GamePhase, Dictionary<string, int>>();
            this.actualMoveFrequenciesByPhase = new Dictionary<GamePhase, Dictionary<string, int>>();
            this.artifacts = new Dictionary<string, int>();
        }

        /// <summary>
        /// Loads the data.
        /// </summary>
        public void LoadData()
        {
            var edgejson = EmbeddedResource.ReadAllJson("ParallelReverseAutoDiff.GnnExample.Statistics", "edge_frequencies_2773067");
            this.edgeFrequenciesByPhase = JsonConvert.DeserializeObject<Dictionary<GamePhase, Dictionary<string, int>>>(edgejson) ?? throw new InvalidOperationException("Could not parse edge JSON");

            var movejson = EmbeddedResource.ReadAllJson("ParallelReverseAutoDiff.GnnExample.Statistics", "move_frequencies_2773067");
            this.moveFrequenciesByPhase = JsonConvert.DeserializeObject<Dictionary<GamePhase, Dictionary<string, int>>>(movejson) ?? throw new InvalidOperationException("Could not parse move JSON");

            var actualmovejson = EmbeddedResource.ReadAllJson("ParallelReverseAutoDiff.GnnExample.Statistics", "actualmove_frequencies_2773067");
            this.actualMoveFrequenciesByPhase = JsonConvert.DeserializeObject<Dictionary<GamePhase, Dictionary<string, int>>>(actualmovejson) ?? throw new InvalidOperationException("Could not parse actual move JSON");

            var artifactsjson = EmbeddedResource.ReadAllJson("ParallelReverseAutoDiff.GnnExample.Statistics", "artifacts");
            this.artifacts = JsonConvert.DeserializeObject<Dictionary<string, int>>(artifactsjson) ?? throw new InvalidOperationException("Could not parse artifacts JSON");

            GapGraph gapGraph = new GapGraph();
            gapGraph.GapNodes = new List<GapNode>();
            gapGraph.GapEdges = new List<GapEdge>();
            gapGraph.GapPaths = new List<GapPath>();
            for (int i = 0; i < 8; ++i)
            {
                for (int j = 0; j < 8; ++j)
                {
                    GapNode node = new GapNode()
                    {
                        Id = Guid.NewGuid(),
                        PositionX = i,
                        PositionY = j,
                    };
                    gapGraph.GapNodes.Add(node);
                }
            }

            var graphJson = JsonConvert.SerializeObject(gapGraph);

            int total = this.loader.GetTotal();
            for (int t = 17066; t < total; ++t)
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
                            var gamePhase = this.gameState.GetGamePhase();
                            var allmoves = this.gameState.GetMoves();
                            var legalmoves = this.gameState.GetAllMoves();
                            var graph = JsonConvert.DeserializeObject<GapGraph>(graphJson) ?? throw new InvalidOperationException("Failed to deserialize.");
                            graph.Populate();
                            graph = this.gameState.PopulateNodes(graph);

                            foreach (var allmove in allmoves)
                            {
                                var path = GameState.GetGapPath(graph, allmove, nextmove, legalmoves);
                                graph.GapPaths.Add(path);
                            }

                            foreach (var legalmove in legalmoves)
                            {
                                var edges = GameState.GetGapEdge(graph, legalmove);
                                graph.GapEdges.Add(edges.Edge1);
                                graph.GapEdges.Add(edges.Edge2);
                            }

                            graph.FormAdjacencyMatrix();
                            var totalStats = 2773067d;
                            graph.UpdateFeatureIndices(this.artifacts, this.gameState.Board.ToPositionFen(), this.gameState.Board.ExecutedMoves.Last().ToString(), gamePhase);
                            graph.UpdateFeatureVectors(this.edgeFrequenciesByPhase[gamePhase], totalStats);
                            graph.UpdateFeatureVectors(this.moveFrequenciesByPhase[gamePhase], totalStats);
                            graph.UpdateFeatureVectors(this.actualMoveFrequenciesByPhase[gamePhase], totalStats);
                            graph.UpdateFeatureVectors();
                            var gJson = JsonConvert.SerializeObject(graph);
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
                    foreach (var buffer in buffers)
                    {
                        Guid guid = Guid.NewGuid();
                        ZipArchiveEntry entry = archive.CreateEntry($"{guid}.json");
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
