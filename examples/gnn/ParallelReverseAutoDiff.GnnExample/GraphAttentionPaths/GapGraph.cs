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
