//------------------------------------------------------------------------------
// <copyright file="GapEdge.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using System.Text.Json.Serialization;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// An edge of the GAP graph.
    /// </summary>
    [Serializable]
    public class GapEdge : IPopulate
    {
        private Guid nodeId;

        /// <summary>
        /// Gets or sets the unique identifier of the edge.
        /// </summary>
        public Guid Id { get; set; }

        /// <summary>
        /// Gets or sets the node that the edge connects to.
        /// </summary>
        [JsonIgnore]
        public GapNode Node { get; set; }

        /// <summary>
        /// Gets or sets the node ID.
        /// </summary>
        public Guid NodeId
        {
            get { return this.Node?.Id ?? Guid.Empty; }
            set { this.nodeId = value; } // Setter for deserialization
        }

        /// <summary>
        /// Gets or sets the feature vector of the edge.
        /// </summary>
        public Matrix FeatureVector { get; set; }

        /// <summary>
        /// Gets or sets the features.
        /// </summary>
        public List<double> Features { get; set; } = new List<double>();

        /// <summary>
        /// Gets or sets the feature indices.
        /// </summary>
        public List<int> FeatureIndices { get; set; } = new List<int>();

        /// <summary>
        /// Populates the node with the node ID.
        /// </summary>
        /// <param name="graph">The graph.</param>
        public void Populate(GapGraph graph)
        {
            var node = graph.GapNodes.FirstOrDefault(n => n.Id == this.nodeId);
            if (node != null)
            {
                this.Node = node;
            }
            else
            {
                throw new InvalidOperationException($"Node with id {this.nodeId} not found.");
            }
        }
    }
}
