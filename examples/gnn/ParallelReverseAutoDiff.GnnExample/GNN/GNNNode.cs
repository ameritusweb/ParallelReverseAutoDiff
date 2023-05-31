//------------------------------------------------------------------------------
// <copyright file="GNNNode.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample.GNN
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// The GNN Node.
    /// </summary>
    public class GNNNode
    {
        /// <summary>
        /// Gets or sets the node type.
        /// </summary>
        public int Type { get; set; }

        /// <summary>
        /// Gets or sets the node position X.
        /// </summary>
        public int X { get; set; }

        /// <summary>
        /// Gets or sets the node position Y.
        /// </summary>
        public int Y { get; set; }

        /// <summary>
        /// Gets or sets the node state.
        /// </summary>
        public Matrix State { get; set; }

        /// <summary>
        /// Gets or sets the edges.
        /// </summary>
        public List<GNNEdge> Edges { get; set; }

        /// <summary>
        /// Gets or sets the messages from one hop away.
        /// </summary>
        public Matrix Messages { get; set; }

        /// <summary>
        /// Gets or sets the messages from two hops away.
        /// </summary>
        public Matrix MessagesTwoHops { get; set; }
    }
}
