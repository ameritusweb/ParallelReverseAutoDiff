//------------------------------------------------------------------------------
// <copyright file="GapNode.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using System.Text.Json.Serialization;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A node in a graph attention path.
    /// </summary>
    [Serializable]
    public class GapNode : IPopulate
    {
        private List<Guid> edgeIds;

        /// <summary>
        /// Initializes a new instance of the <see cref="GapNode"/> class.
        /// </summary>
        public GapNode()
        {
            this.FeatureVector = new Matrix(1, 1);
            this.Edges = new List<GapEdge>();
        }

        /// <summary>
        /// Gets or sets the node identifier.
        /// </summary>
        public Guid Id { get; set; }

        /// <summary>
        /// Gets or sets the node position X in the graph.
        /// </summary>
        public int PositionX { get; set; }

        /// <summary>
        /// Gets or sets the node position Y in the graph.
        /// </summary>
        public int PositionY { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether the node is in the path.
        /// </summary>
        public bool IsInPath { get; set; }

        /// <summary>
        /// Gets or sets a tag.
        /// </summary>
        public object Tag { get; set; }

        /// <summary>
        /// Gets or sets the type of node.
        /// </summary>
        public GapType GapType { get; set; }

        /// <summary>
        /// Gets or sets the feature vector of the node.
        /// </summary>
        public Matrix FeatureVector { get; set; }

        /// <summary>
        /// Gets or sets the edges.
        /// </summary>
        [JsonIgnore]
        public List<GapEdge> Edges { get; set; }

        /// <summary>
        /// Gets or sets the edge identifiers.
        /// </summary>
        public List<Guid> EdgeIds
        {
            get { return (this.Edges?.Select(e => e.Id) ?? new List<Guid>()).ToList(); }
            set { this.edgeIds = value; } // Setter for deserialization
        }

        /// <summary>
        /// Populate the edges of the node.
        /// </summary>
        /// <param name="graph">The graph.</param>
        public void Populate(GapGraph graph)
        {
            if (this.edgeIds != null)
            {
                var edges = this.edgeIds.Select(id => graph.GapEdges.FirstOrDefault(e => e.Id == id)).ToList();
                foreach (var edge in edges)
                {
                    if (edge != null)
                    {
                        this.Edges.Add(edge);
                    }
                    else
                    {
                        throw new InvalidOperationException("Edge not found.");
                    }
                }
            }
        }
    }
}
