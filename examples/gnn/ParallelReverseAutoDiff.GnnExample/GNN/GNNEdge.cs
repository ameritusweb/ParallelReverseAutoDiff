//------------------------------------------------------------------------------
// <copyright file="GNNEdge.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample.GNN
{
    using Chess;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A GNN Edge.
    /// </summary>
    public class GNNEdge
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GNNEdge"/> class.
        /// </summary>
        /// <param name="from">The from node.</param>
        /// <param name="to">The to node.</param>
        public GNNEdge(GNNNode from, GNNNode to)
        {
            this.From = from;
            this.To = to;
        }

        /// <summary>
        /// Gets or sets the node from.
        /// </summary>
        public GNNNode From { get; set; }

        /// <summary>
        /// Gets or sets the node to.
        /// </summary>
        public GNNNode To { get; set; }

        /// <summary>
        /// Gets or sets the move type.
        /// </summary>
        public MoveType MoveType { get; set; }

        /// <summary>
        /// Gets or sets the capture piece type.
        /// </summary>
        public char? CapturePieceType { get; set; }

        /// <summary>
        /// Gets or sets the promotion piece type.
        /// </summary>
        public char? PromotionPieceType { get; set; }


        public char PieceType { get; set; }

        /// <summary>
        /// Gets or sets the node state.
        /// </summary>
        public Matrix State { get; set; }
    }
}
