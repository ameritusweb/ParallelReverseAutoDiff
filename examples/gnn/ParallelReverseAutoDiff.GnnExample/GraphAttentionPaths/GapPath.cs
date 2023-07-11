//------------------------------------------------------------------------------
// <copyright file="GapPath.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// The GAP path.
    /// </summary>
    [Serializable]
    public class GapPath
    {
        private List<Guid> nodeIds;

        /// <summary>
        /// Initializes a new instance of the <see cref="GapPath"/> class.
        /// </summary>
        public GapPath()
        {
            this.FeatureVector = new Matrix(1, 1);
        }

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
        [Newtonsoft.Json.JsonIgnore]
        public List<GapNode> Nodes { get; set; } = new List<GapNode>();

        /// <summary>
        /// Gets or sets the node IDs of the path.
        /// </summary>
        [Newtonsoft.Json.JsonProperty]
        public List<Guid> NodeIds
        {
            get
            {
                if (this.Nodes == null || !this.Nodes.Any())
                {
                    return this.nodeIds;
                }
                else
                {
                    return this.Nodes.Select(e => e.Id).ToList();
                }
            }

            set
            {
                this.nodeIds = value;
            } // Setter for deserialization
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
        /// Gets the move.
        /// </summary>
        /// <returns>The move.</returns>
        public string Move()
        {
            var firstNode = this.Nodes[0];
            var lastNode = this.Nodes[this.Nodes.Count - 1];
            return $"{this.GapType.ToString()} {this.File(firstNode)}{firstNode.PositionY + 1} - {this.File(lastNode)}{lastNode.PositionY + 1}";
        }

        /// <summary>
        /// Populates the nodes of the path based on the node IDs.
        /// </summary>
        /// <param name="graph">The graph.</param>
        public void Populate(GapGraph graph)
        {
            if (this.nodeIds != null)
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

            this.AdjacencyIndex = graph.GapPaths.IndexOf(this);
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

        private string File(GapNode node)
        {
            switch (node.PositionX)
            {
                case 0:
                    return "A";
                case 1:
                    return "B";
                case 2:
                    return "C";
                case 3:
                    return "D";
                case 4:
                    return "E";
                case 5:
                    return "F";
                case 6:
                    return "G";
                case 7:
                    return "H";
                default:
                    return string.Empty;
            }
        }
    }
}
