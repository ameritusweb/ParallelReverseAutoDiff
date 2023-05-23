using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GnnExample.GNN
{
    public class GNNWeightedGraph
    {
        public List<GNNNode> Nodes { get; set; }
        public List<GNNEdge> Edges { get; set; }
    }
}
