//------------------------------------------------------------------------------
// <copyright file="TrainingSetGenerator.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
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

            var moves = this.loader.LoadMoves(0);
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
                        var path = GameState.GetGapPath(graph, allmove, nextmove);
                        graph.GapPaths.Add(path);
                    }

                    foreach (var legalmove in legalmoves)
                    {
                        var edges = GameState.GetGapEdge(graph, legalmove);
                        graph.GapEdges.Add(edges.Edge1);
                        graph.GapEdges.Add(edges.Edge2);
                    }
                }
            }
        }
    }
}
