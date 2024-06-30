//------------------------------------------------------------------------------
// <copyright file="PradEdge.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    /// <summary>
    /// An edge for a PRAD graph.
    /// </summary>
    public class PradEdge
    {
        /// <summary>
        /// Gets or sets the from node.
        /// </summary>
        public PradNode From { get; set; }

        /// <summary>
        /// Gets or sets the to node.
        /// </summary>
        public PradNode To { get; set; }
    }
}
