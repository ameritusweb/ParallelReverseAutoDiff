//------------------------------------------------------------------------------
// <copyright file="GapType.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using ManagedCuda.BasicTypes;
    using ParallelReverseAutoDiff.RMAD;
    using System.Text.Json.Serialization;

    /// <summary>
    /// The GAP path.
    /// </summary>
    [Serializable]
    public class GapPath
    {
        /// <summary>
        /// An identifier for the path.
        /// </summary>
        public Guid Id { get; set; }

        /// <summary>
        /// An indicator of whether the path is the target.
        /// </summary>
        public bool IsTarget { get; set; }

        /// <summary>
        /// The index of the adjacency matrix.
        /// </summary>
        public int AdjacencyIndex { get; set; }

        /// <summary>
        /// The nodes of the path.
        /// </summary>
        [JsonIgnore]
        public List<GapNode> Nodes { get; set; }

        private List<Guid> nodeIds;
        public List<Guid> NodeIds
        {
            get { return (Nodes?.Select(e => e.Id) ?? new List<Guid>()).ToList(); }
            set { nodeIds = value; }  // Setter for deserialization
        }

        public void Populate(GapGraph graph)
        {
            var nodes = nodeIds.Select(id => graph.GapNodes.FirstOrDefault(n => n.Id == id)).ToList();
            foreach (var node in nodes)
            {
                if (node != null)
                {
                    this.Nodes.Add(node);
                }
                else
                {
                    throw new InvalidOperationException("Node not found in graph.");
                }
            }
        }

        /// <summary>
        /// The feature vector of the path.
        /// </summary>
        public Matrix FeatureVector { get; set; }

        /// <summary>
        /// Add a node to the path.
        /// </summary>
        /// <param name="node">The added node.</param>
        public void AddNode(GapNode node)
        {
            this.Nodes.Add(node);
            node.IsInPath = true;
        }

        /// <summary>
        /// The type of the path.
        /// </summary>
        public GapType GapType
        {
            get
            {
                return this.Nodes[0].GapType;
            }
        }
    }
}
