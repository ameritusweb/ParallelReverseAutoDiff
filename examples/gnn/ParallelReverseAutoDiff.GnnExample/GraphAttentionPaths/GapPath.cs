using ParallelReverseAutoDiff.RMAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    public class GapPath
    {
        public Guid Id { get; set; }

        public bool IsTarget { get; set; }

        public int AdjacencyIndex { get; set; }

        public List<GapNode> Nodes { get; set; }

        public Matrix FeatureVector { get; set; }

        public void AddNode(GapNode node)
        {
            this.Nodes.Add(node);
            node.IsInPath = true;
        }

        public GapType GapType
        {
            get
            {
                return this.Nodes[0].GapType;
            }
        }
    }
}
