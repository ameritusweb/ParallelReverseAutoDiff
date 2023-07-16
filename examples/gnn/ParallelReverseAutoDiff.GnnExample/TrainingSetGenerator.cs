//------------------------------------------------------------------------------
// <copyright file="TrainingSetGenerator.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using System.Collections.Concurrent;
    using System.IO.Compression;
    using System.Xml.Linq;
    using Chess;
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.GnnExample.Common;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths;
    using static ILGPU.IR.Transformations.CodePlacement;

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

        private string graphJson;

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
        /// Initialize the training set generator.
        /// </summary>
        public void Initialize()
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

            this.graphJson = JsonConvert.SerializeObject(gapGraph);
        }

        /// <summary>
        /// Add to bag of graphs.
        /// </summary>
        /// <param name="bagOfGraphs">The graphs.</param>
        /// <param name="rand">The random.</param>
        /// <param name="numberOfMoves">The number of moves.</param>
        /// <returns>The task.</returns>
        public async Task AddToBag(ConcurrentBag<GapGraph> bagOfGraphs, Random rand, int numberOfMoves)
        {
            await Task.Run(() =>
            {
                int total = this.loader.GetTotal();
                var r = rand.Next(total);
                var moves = this.loader.LoadMoves(r);
                var name = this.loader.GetFileName(r).Replace(".pgn", string.Empty);
                var graphs = this.ProcessMoves(moves.Take(numberOfMoves).ToList(), name, false);
                var randomGraphs = graphs.OrderBy(x => rand.Next());
                randomGraphs.ToList().ForEach(x => bagOfGraphs.Add(x));
            });
        }

        /// <summary>
        /// Loads the data.
        /// </summary>
        public void LoadData()
        {
            this.Initialize();

            int total = this.loader.GetTotal();
            for (int t = 0; t < total; ++t)
            {
                var moves = this.loader.LoadMoves(t);
                var name = this.loader.GetFileName(t).Replace(".pgn", string.Empty);
                this.ProcessMoves(moves, name, true);
            }
        }

        private List<GapGraph> ProcessMoves(List<Move> moves, string name, bool shouldSave)
        {
            this.gameState = new GameState();
            List<string> jsons = new List<string>();
            List<GapGraph> graphs = new List<GapGraph>();
            try
            {
                foreach ((Move move, Move? nextmove) in moves.WithNext())
                {
                    this.gameState.Board.Move(move);
                    if (nextmove != null)
                    {
                        var gamePhase = this.gameState.GetGamePhase();
                        var allmoves = this.gameState.GetMoves();
                        var legalbothmoves = this.gameState.GetAllMoves();
                        var legalmoves = this.gameState.Board.Moves().ToList();
                        var turn = this.gameState.Board.Turn;
                        var fen = this.gameState.Board.ToFen();
                        var graph = JsonConvert.DeserializeObject<GapGraph>(this.graphJson) ?? throw new InvalidOperationException("Failed to deserialize.");
                        graph.Id = Guid.NewGuid();
                        graph.Populate();
                        graph = this.gameState.PopulateNodes(graph);
                        graph.FenString = fen;

                        foreach (var allmove in allmoves)
                        {
                            var path = GameState.GetGapPath(graph, allmove, nextmove, legalbothmoves, legalmoves, turn);
                            graph.GapPaths.Add(path);
                        }

                        foreach (var legalmove in legalbothmoves)
                        {
                            var edges = GameState.GetGapEdge(graph, legalmove, legalmoves, turn);
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

                        foreach (var path in graph.GapPaths)
                        {
                            var firstNode = path.Nodes.First();
                            var associatedEdge = firstNode.Edges.First(x => x.Move().StartsWith(path.MoveString));
                            path.EdgeId = associatedEdge.Id;
                        }

                        if (graph.GapPaths.Count(x => x.IsTarget) > 1
                            ||
                            !graph.GapPaths.Any(x => x.IsTarget))
                        {
                            var graphTargets = graph.GapPaths.Count(x => x.IsTarget);
                            continue;
                        }

                        graphs.Add(graph);
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

            if (shouldSave)
            {
                this.SaveToZip(jsons, $"E:\\graphs\\{name}.zip");
            }

            return graphs;
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
