//------------------------------------------------------------------------------
// <copyright file="GNNMove.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample.GNN
{
    /// <summary>
    /// The GNN selected move.
    /// </summary>
    public class GNNMove
    {
        /// <summary>
        /// Gets or sets the node from.
        /// </summary>
        public GNNNode NodeFrom { get; set; }

        /// <summary>
        /// Gets or sets the node to.
        /// </summary>
        public GNNNode NodeTo { get; set; }

        /// <summary>
        /// Gets or sets the score for the move.
        /// </summary>
        public double Score { get; set; }
    }
}
