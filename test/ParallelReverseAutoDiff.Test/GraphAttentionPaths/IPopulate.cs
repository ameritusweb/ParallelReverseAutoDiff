//------------------------------------------------------------------------------
// <copyright file="IPopulate.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    /// <summary>
    /// The public interface for populating after deserialization.
    /// </summary>
    public interface IPopulate
    {
        /// <summary>
        /// Populates the graph after deserialization.
        /// </summary>
        /// <param name="graph">The graph.</param>
        public void Populate(GapGraph graph);
    }
}
