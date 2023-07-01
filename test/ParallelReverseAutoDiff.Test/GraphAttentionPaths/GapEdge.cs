//------------------------------------------------------------------------------
// <copyright file="GapGraph.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// An edge of the GAP graph.
    /// </summary>
    public class GapEdge
    {
        /// <summary>
        /// The node that the edge connects to.
        /// </summary>
        public GapNode Node { get; set; }

        /// <summary>
        /// The feature vector of the edge.
        /// </summary>
        public Matrix FeatureVector { get; set; }
    }
}
