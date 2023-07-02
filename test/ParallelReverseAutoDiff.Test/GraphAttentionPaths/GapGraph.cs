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
    }
}
