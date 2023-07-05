//------------------------------------------------------------------------------
// <copyright file="GapGraph.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using ParallelReverseAutoDiff.GnnExample;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A graph attention paths graph.
    /// </summary>
    [Serializable]
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
        [Newtonsoft.Json.JsonIgnore]
        public List<GapPath> TargetPaths
        {
            get
            {
                return this.GapPaths.Where(x => x.IsTarget).ToList();
            }
        }

        /// <summary>
        /// Update the feature indices.
        /// </summary>
        /// <param name="artifacts">The artifacts.</param>
        /// <param name="fen">The fen.</param>
        /// <param name="lastMove">The last move.</param>
        /// <param name="gamePhase">The game phase.</param>
        public void UpdateFeatureIndices(Dictionary<string, int> artifacts, string fen, string lastMove, GamePhase gamePhase)
        {
            lastMove = lastMove.ToLowerInvariant().Replace("o-o-o", "o3").Replace("o-o", "o2");
            foreach (var edge in this.GapEdges)
            {
                if (gamePhase == GamePhase.Opening)
                {
                    edge.FeatureIndices.Add(artifacts["opening"]);
                }
                else if (gamePhase == GamePhase.MiddleGame)
                {
                    edge.FeatureIndices.Add(artifacts["middlegame"]);
                }
                else
                {
                    edge.FeatureIndices.Add(artifacts["endgame"]);
                }

                dynamic tag = edge.Tag;
                bool start = tag.Start;
                string move = tag.Move;
                var fens = fen.Split(' ').ToList();
                for (int i = 1; i < fens.Count; i++)
                {
                    if (artifacts.ContainsKey(fens[i]))
                    {
                        edge.FeatureIndices.Add(artifacts[fens[i]]);
                    }
                    else
                    {
                        edge.FeatureIndices.Add(artifacts[string.Empty]);
                    }
                }

                lastMove = lastMove.Trim(new[] { '{', '}' });
                string[] lastmoves = lastMove.Split('-', StringSplitOptions.RemoveEmptyEntries);
                edge.FeatureIndices.Add(artifacts[lastmoves[0].Trim()]);
                edge.FeatureIndices.Add(artifacts[lastmoves[1].Trim()]);
                edge.FeatureIndices.Add(artifacts[lastmoves[2].Trim()]);
                if (lastmoves.Length > 3)
                {
                    edge.FeatureIndices.Add(artifacts[lastmoves[3].Trim()]);
                }
                else
                {
                    edge.FeatureIndices.Add(artifacts["nocapture"]);
                }

                if (lastmoves.Length > 4)
                {
                    edge.FeatureIndices.Add(artifacts[lastmoves[4].Trim()]);
                }
                else
                {
                    edge.FeatureIndices.Add(artifacts["noparameter"]);
                }

                if (move.StartsWith("{"))
                {
                    move = move.Trim(new[] { '{', '}' });
                    string[] moves = move.Split('-', StringSplitOptions.RemoveEmptyEntries);
                    edge.FeatureIndices.Add(start ? artifacts["start"] : artifacts["nonstart"]);
                    edge.FeatureIndices.Add(artifacts[moves[0].Trim()]);
                    edge.FeatureIndices.Add(artifacts[moves[1].Trim()]);
                    edge.FeatureIndices.Add(artifacts[moves[2].Trim()]);
                    if (moves.Length > 3)
                    {
                        edge.FeatureIndices.Add(artifacts[moves[3].Trim()]);
                    }
                    else
                    {
                        edge.FeatureIndices.Add(artifacts["nocapture"]);
                    }

                    if (moves.Length > 4)
                    {
                        edge.FeatureIndices.Add(artifacts[moves[4].Trim()]);
                    }
                    else
                    {
                        edge.FeatureIndices.Add(artifacts["noparameter"]);
                    }

                    edge.FeatureIndices.Add(artifacts["nodefense"]);
                }
                else
                {
                    edge.FeatureIndices.Add(start ? artifacts["start"] : artifacts["nonstart"]);
                    edge.FeatureIndices.Add(artifacts[move.Substring(0, 2).ToLowerInvariant()]);
                    edge.FeatureIndices.Add(artifacts[move.Substring(2, 2).ToLowerInvariant()]);
                    edge.FeatureIndices.Add(artifacts[move.Substring(4, 2).ToLowerInvariant()]);
                    edge.FeatureIndices.Add(artifacts[move.Length < 8 ? "nocapture" : move.Substring(6, 2).ToLowerInvariant()]);
                    edge.FeatureIndices.Add(artifacts["noparameter"]);
                    if (move.Length >= 8)
                    {
                        if (move.Substring(0, 1) == move.Substring(6, 1))
                        {
                            edge.FeatureIndices.Add(artifacts["defense"]);
                        }
                        else
                        {
                            edge.FeatureIndices.Add(artifacts["nodefense"]);
                        }
                    }
                    else
                    {
                        edge.FeatureIndices.Add(artifacts["nodefense"]);
                    }
                }

                if (edge.FeatureIndices.Count != 16)
                {
                }
            }
        }

        /// <summary>
        /// Update feature vectors.
        /// </summary>
        public void UpdateFeatureVectors()
        {
            foreach (var edge in this.GapEdges)
            {
                edge.FeatureVector = new Matrix(edge.FeatureIndices.Count, 1);
                for (int i = 0; i < edge.FeatureIndices.Count; i++)
                {
                    edge.FeatureVector[i, 0] = edge.FeatureIndices[i];
                }
            }
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
                string move = tag.Move;
                if (map.ContainsKey(move))
                {
                    var stat = (double)map[move];
                    edge.Features.Add(stat / totalStats);
                }
                else
                {
                    if (move.StartsWith("{"))
                    {
                        move = move.Trim(new[] { '{', '}' });
                        string[] moves = move.Split('-', StringSplitOptions.RemoveEmptyEntries);
                        move = string.Join(string.Empty, moves.Select(x => x.Trim()));

                        if (map.ContainsKey(move))
                        {
                            var stat = (double)map[move];
                            edge.Features.Add(stat / totalStats);
                            continue;
                        }
                        else
                        {
                            edge.Features.Add(0d);
                        }
                    }
                    else
                    {
                        var keys = map.Keys.ToList();
                        if (move.Length == 6)
                        {
                            move = "{" + move.Substring(0, 2) + " - " + move.Substring(2, 2) + " - " + move.Substring(4, 2);
                            if (keys.Any(x => x.StartsWith(move)))
                            {
                                var stats = map.Where(x => x.Key.StartsWith(move, StringComparison.OrdinalIgnoreCase)).ToList();
                                var stat = stats.Sum(x => x.Value);
                                edge.Features.Add(stat / totalStats);
                                continue;
                            }
                        }
                        else if (move.Length == 8)
                        {
                            move = "{" + move.Substring(0, 2) + " - " + move.Substring(2, 2) + " - " + move.Substring(4, 2) + " - " + move.Substring(6, 2);
                            if (keys.Any(x => x.StartsWith(move)))
                            {
                                var stats = map.Where(x => x.Key.StartsWith(move, StringComparison.OrdinalIgnoreCase)).ToList();
                                var stat = stats.Sum(x => x.Value);
                                edge.Features.Add(stat / totalStats);
                                continue;
                            }
                        }
                        else
                        {
                        }

                        edge.Features.Add(0d);
                    }
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
                    var path1 = this.GapPaths[i];
                    var path2 = this.GapPaths[j];
                    path1.AdjacencyIndex = i;
                    path2.AdjacencyIndex = j;
                    var intersection = path1.Nodes.Select(x => x.ToString()).Intersect(path2.Nodes.Select(x => x.ToString()));
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
            foreach (var edge in this.GapEdges)
            {
                edge.Populate(this);
            }

            foreach (var path in this.GapPaths)
            {
                path.Populate(this);
            }

            foreach (var node in this.GapNodes)
            {
                node.Populate(this);
            }
        }
    }
}
