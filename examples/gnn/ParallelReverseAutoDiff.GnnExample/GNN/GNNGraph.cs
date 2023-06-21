//------------------------------------------------------------------------------
// <copyright file="GNNGraph.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample.GNN
{
    /// <summary>
    /// The GNN graph.
    /// </summary>
    public class GNNGraph
    {
        /// <summary>
        /// Gets or sets the GNN graph nodes.
        /// </summary>
        public List<GNNNode> Nodes { get; set; } = new List<GNNNode>();

        /// <summary>
        /// Gets or sets the GNN graph edges.
        /// </summary>
        public List<GNNEdge> Edges { get; set; } = new List<GNNEdge>();
    }
}
