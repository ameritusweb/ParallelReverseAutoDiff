//------------------------------------------------------------------------------
// <copyright file="Node.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.TicTacToe
{
    /// <summary>
    /// A node in the MCTS tree.
    /// </summary>
    public class Node
    {
        /// <summary>
        /// Gets or sets the state of the board.
        /// </summary>
        public TicTacToeBoard State { get; set; }

        /// <summary>
        /// Gets or sets the parent node.
        /// </summary>
        public Node Parent { get; set; }

        /// <summary>
        /// Gets or sets the children.
        /// </summary>
        public List<Node> Children { get; set; }

        /// <summary>
        /// Gets or sets the number of visits.
        /// </summary>
        public int Visits { get; set; }

        /// <summary>
        /// Gets or sets the MCTS value.
        /// </summary>
        public double Value { get; set; }
    }
}
