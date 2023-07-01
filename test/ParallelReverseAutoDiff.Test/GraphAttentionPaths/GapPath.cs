//------------------------------------------------------------------------------
// <copyright file="GapType.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// The GAP path.
    /// </summary>
    public class GapPath
    {
        /// <summary>
        /// An identifier for the path.
        /// </summary>
        public Guid Id { get; set; }

        /// <summary>
        /// An indicator of whether the path is the target.
        /// </summary>
        public bool IsTarget { get; set; }

        /// <summary>
        /// The index of the adjacency matrix.
        /// </summary>
        public int AdjacencyIndex { get; set; }

        /// <summary>
        /// The nodes of the path.
        /// </summary>
        public List<GapNode> Nodes { get; set; }

        /// <summary>
        /// The feature vector of the path.
        /// </summary>
        public Matrix FeatureVector { get; set; }

        /// <summary>
        /// Add a node to the path.
        /// </summary>
        /// <param name="node">The added node.</param>
        public void AddNode(GapNode node)
        {
            this.Nodes.Add(node);
            node.IsInPath = true;
        }

        /// <summary>
        /// The type of the path.
        /// </summary>
        public GapType GapType
        {
            get
            {
                return this.Nodes[0].GapType;
            }
        }
    }
}
