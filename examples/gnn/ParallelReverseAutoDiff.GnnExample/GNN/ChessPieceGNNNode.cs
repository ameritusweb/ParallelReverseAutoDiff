//------------------------------------------------------------------------------
// <copyright file="ChessPieceGNNNode.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample.GNN
{
    using Chess;

    /// <summary>
    /// A chess piece GNN Node.
    /// </summary>
    public class ChessPieceGNNNode : GNNNode
    {
        /// <summary>
        /// Gets or sets the piece type.
        /// </summary>
        public PieceType PieceType { get; set; }

        /// <summary>
        /// Gets or sets the piece color.
        /// </summary>
        public PieceColor PieceColor { get; set; }

        /// <summary>
        /// Gets or sets the piece position.
        /// </summary>
        public Position Position { get; set; }
    }
}
