//------------------------------------------------------------------------------
// <copyright file="GapGraph.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A graph attention paths graph.
    /// </summary>
    public class GapGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GapGraph"/> class.
        /// </summary>
        public GapGraph()
        {
            this.AdjacencyMatrix = new Matrix(1, 1);
            this.NormalizedAdjacency = new Matrix(1, 1);
        }

        /// <summary>
        /// Gets or sets the edges of the graph.
        /// </summary>
        public List<GapEdge> GapEdges { get; set; }

        /// <summary>
        /// Gets or sets the nodes of the graph.
        /// </summary>
        public List<GapNode> GapNodes { get; set; }

        /// <summary>
        /// Gets or sets the paths of the graph.
        /// </summary>
        public List<GapPath> GapPaths { get; set; }

        /// <summary>
        /// Gets or sets the adjacency matrix of the graph.
        /// </summary>
        public Matrix AdjacencyMatrix { get; set; }

        /// <summary>
        /// Gets or sets the normalized adjacency matrix of the graph.
        /// </summary>
        public Matrix NormalizedAdjacency { get; set; }

        /// <summary>
        /// Gets the target paths.
        /// </summary>
        public List<GapPath> TargetPaths
        {
            get
            {
                return this.GapPaths.Where(x => x.IsTarget).ToList();
            }
        }

        /// <summary>
        /// Add features by piece and edge.
        /// </summary>
        /// <param name="piece">The piece.</param>
        /// <param name="edge">The edge.</param>
        public void AddFeaturesByPiece(string piece, GapEdge edge)
        {
            string[] pieces = new[] { "wq", "wr", "wb", "wn", "wp", "wk", "bq", "br", "bb", "bn", "bp", "bk" };
            foreach (var pieceExample in pieces)
            {
                if (piece.ToLower() == pieceExample)
                {
                    edge.Features.Add(1.0d);
                }
                else
                {
                    edge.Features.Add(0.0d);
                }
            }
        }

        /// <summary>
        /// Add features by square and edge.
        /// </summary>
        /// <param name="square">The square.</param>
        /// <param name="edge">The edge.</param>
        public void AddFeaturesBySquare(string square, GapEdge edge)
        {
            string[] squares = new[]
            {
                "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8",
                "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8",
                "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8",
                "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8",
                "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8",
                "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8",
                "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8",
                "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8",
            };
    }

        /// <summary>
        /// Update the feature vectors.
        /// </summary>
        /// <param name="map">The statistical map.</param>
        /// <param name="totalStats">The total statistics.</param>
        public void UpdateFeatureVectors(Dictionary<string, int> map, double totalStats)
        {
            foreach (var edge in this.GapEdges)
            {
                dynamic tag = edge.Tag;
                bool start = tag.Start;
                string move = tag.Move;
                this.AddFeaturesByPiece(move.Substring(0, 2), edge);
                this.AddFeaturesBySquare(move.Substring(2, 2), edge);
                this.AddFeaturesBySquare(move.Substring(4, 2), edge);
                this.AddFeaturesByPiece(move.Length < 8 ? string.Empty : move.Substring(6, 2), edge);
                edge.Features.Add(start ? 1.0d : 0.0d);
                if (map.ContainsKey(move))
                {
                    var stat = (double)map[move];
                    edge.Features.Add(stat / totalStats);
                }
                else
                {
                    edge.Features.Add(0d);
                }
            }
        }

        /// <summary>
        /// Forms the adjacency matrix.
        /// </summary>
        public void FormAdjacencyMatrix()
        {
            this.AdjacencyMatrix = new Matrix(this.GapPaths.Count, this.GapPaths.Count);
            for (int i = 0; i < this.GapPaths.Count; i++)
            {
                for (int j = 0; j < this.GapPaths.Count; j++)
                {
                    var intersection = this.GapPaths[i].Nodes.Select(x => x.ToString()).Intersect(this.GapPaths[j].Nodes.Select(x => x.ToString()));
                    if (intersection.Any())
                    {
                        this.AdjacencyMatrix[i, j] = 1d;
                    }
                }
            }
        }

        /// <summary>
        /// Populate after deserialization.
        /// </summary>
        public void Populate()
        {
            foreach (var node in this.GapNodes)
            {
                node.Populate(this);
            }

            foreach (var edge in this.GapEdges)
            {
                edge.Populate(this);
            }

            foreach (var path in this.GapPaths)
            {
                path.Populate(this);
            }
        }
    }
}
