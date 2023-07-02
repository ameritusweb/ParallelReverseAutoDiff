//------------------------------------------------------------------------------
// <copyright file="GapGraph.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using ManagedCuda.BasicTypes;
    using ParallelReverseAutoDiff.RMAD;
    using System.Text.Json.Serialization;

    /// <summary>
    /// An edge of the GAP graph.
    /// </summary>
    [Serializable]
    public class GapEdge : IPopulate
    {
        /// <summary>
        /// The unique identifier of the edge.
        /// </summary>
        public Guid Id { get; set; }

        /// <summary>
        /// The node that the edge connects to.
        /// </summary>
        [JsonIgnore]
        public GapNode Node { get; set; }

        private Guid nodeId;
        public Guid NodeId
        {
            get { return Node?.Id ?? Guid.Empty; }
            set { nodeId = value; }  // Setter for deserialization
        }

        /// <summary>
        /// The feature vector of the edge.
        /// </summary>
        public Matrix FeatureVector { get; set; }

        public void Populate(GapGraph graph)
        {
            var node = graph.GapNodes.FirstOrDefault(n => n.Id == nodeId);
            if (node != null)
            {
                Node = node;     
            }
            else
            {
                throw new InvalidOperationException($"Node with id {nodeId} not found.");
            }
        }
    }
}
