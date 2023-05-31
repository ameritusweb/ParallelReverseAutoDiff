//------------------------------------------------------------------------------
// <copyright file="GNNEdge.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample.GNN
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A GNN Edge.
    /// </summary>
    public class GNNEdge
    {
        /// <summary>
        /// Gets or sets the node from.
        /// </summary>
        public GNNNode From { get; set; }

        /// <summary>
        /// Gets or sets the node to.
        /// </summary>
        public GNNNode To { get; set; }

        /// <summary>
        /// Gets or sets the node state.
        /// </summary>
        public Matrix State { get; set; }
    }
}
