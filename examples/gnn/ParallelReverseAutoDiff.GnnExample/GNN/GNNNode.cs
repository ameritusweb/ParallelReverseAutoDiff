using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GnnExample.GNN
{
    public class GNNNode
    {
        public int Type { get; set; }
        public int X { get; set; }
        public int Y { get; set; }
        public double[][] State { get; set; }
        public List<GNNEdge> Edges { get; set; }
        public double[][] Messages { get; set; }
        public double[][] MessagesTwoHops { get; set; }
    }
}
