//------------------------------------------------------------------------------
// <copyright file="GapNode.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A node in a graph attention path.
    /// </summary>
    public class GapNode
    {
        /// <summary>
        /// The node identifier.
        /// </summary>
        public Guid Id { get; set; }

        /// <summary>
        /// The node position X in the graph.
        /// </summary>
        public int PositionX { get; set; }

        /// <summary>
        /// The node position Y in the graph.
        /// </summary>
        public int PositionY { get; set; }

        /// <summary>
        /// A value indicating whether the node is in the path.
        /// </summary>
        public bool IsInPath { get; set; }

        /// <summary>
        /// The type of node.
        /// </summary>
        public GapType GapType { get; set; }

        /// <summary>
        /// The feature vector of the node.
        /// </summary>
        public Matrix FeatureVector { get; set; }

        /// <summary>
        /// The edges.
        /// </summary>
        public List<GapEdge> Edges { get; set; }
    }
}
