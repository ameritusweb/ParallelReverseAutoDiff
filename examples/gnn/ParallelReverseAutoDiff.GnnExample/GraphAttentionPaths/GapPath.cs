//------------------------------------------------------------------------------
// <copyright file="GapPath.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using System.Text.Json.Serialization;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// The GAP path.
    /// </summary>
    [Serializable]
    public class GapPath
    {
        private List<Guid> nodeIds;

        /// <summary>
        /// Gets or sets an identifier for the path.
        /// </summary>
        public Guid Id { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether the path is the target.
        /// </summary>
        public bool IsTarget { get; set; }

        /// <summary>
        /// Gets or sets the index of the adjacency matrix.
        /// </summary>
        public int AdjacencyIndex { get; set; }

        /// <summary>
        /// Gets or sets the nodes of the path.
        /// </summary>
        [JsonIgnore]
        public List<GapNode> Nodes { get; set; }

        /// <summary>
        /// Gets or sets the node IDs of the path.
        /// </summary>
        public List<Guid> NodeIds
        {
            get { return (this.Nodes?.Select(e => e.Id) ?? new List<Guid>()).ToList(); }
            set { this.nodeIds = value; } // Setter for deserialization
        }

        /// <summary>
        /// Gets or sets the feature vector of the path.
        /// </summary>
        public Matrix FeatureVector { get; set; }

        /// <summary>
        /// Gets the type of the path.
        /// </summary>
        public GapType GapType
        {
            get
            {
                return this.Nodes[0].GapType;
            }
        }

        /// <summary>
        /// Populates the nodes of the path based on the node IDs.
        /// </summary>
        /// <param name="graph">The graph.</param>
        public void Populate(GapGraph graph)
        {
            var nodes = this.nodeIds.Select(id => graph.GapNodes.FirstOrDefault(n => n.Id == id)).ToList();
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
        /// Add a node to the path.
        /// </summary>
        /// <param name="node">The added node.</param>
        public void AddNode(GapNode node)
        {
            this.Nodes.Add(node);
            node.IsInPath = true;
        }
    }
}
