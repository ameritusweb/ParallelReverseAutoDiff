//------------------------------------------------------------------------------
// <copyright file="GapNode.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using ManagedCuda.BasicTypes;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A node in a graph attention path.
    /// </summary>
    public class GapNode : IPopulate
    {
        /// <summary>
        /// The node identifier.
        /// </summary>
        public Guid Id { get; set; }

        /// <summary>
        /// The node position X in the graph.
        /// </summary>
        public int PositionX { get; set; }

        /// <summary>
        /// The node position Y in the graph.
        /// </summary>
        public int PositionY { get; set; }

        /// <summary>
        /// A value indicating whether the node is in the path.
        /// </summary>
        public bool IsInPath { get; set; }

        /// <summary>
        /// The type of node.
        /// </summary>
        public GapType GapType { get; set; }

        /// <summary>
        /// The feature vector of the node.
        /// </summary>
        public Matrix FeatureVector { get; set; }

        /// <summary>
        /// The edges.
        /// </summary>
        public List<GapEdge> Edges { get; set; }

        private List<Guid> edgeIds;
        public List<Guid> EdgeIds
        {
            get { return (Edges?.Select(e => e.Id) ?? new List<Guid>()).ToList(); }
            set { edgeIds = value; }  // Setter for deserialization
        }

        public void Populate(GapGraph graph)
        {
            var edges = edgeIds.Select(id => graph.GapEdges.FirstOrDefault(e => e.Id == id)).ToList();
            foreach (var edge in edges)
            {
                if (edge != null)
                {
                    Edges.Add(edge);
                }
                else
                {
                    throw new InvalidOperationException("Edge not found.");
                }
            }
        }
    }
}
