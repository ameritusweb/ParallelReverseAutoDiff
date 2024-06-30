//------------------------------------------------------------------------------
// <copyright file="PradNode.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    /// <summary>
    /// A node for a PRAD graph.
    /// </summary>
    public class PradNode
    {
        /// <summary>
        /// Gets or sets the label.
        /// </summary>
        public string Label { get; set; }

        /// <summary>
        /// Gets or sets X.
        /// </summary>
        public int X { get; set; }

        /// <summary>
        /// Gets or sets Y.
        /// </summary>
        public int Y { get; set; }

        /// <summary>
        /// Gets or sets the width.
        /// </summary>
        public int Width { get; set; }

        /// <summary>
        /// Gets or sets the height.
        /// </summary>
        public int Height { get; set; }
    }
}
