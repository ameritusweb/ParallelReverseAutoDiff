using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GnnExample.GNN
{
    public class GNNEdge
    {
        public GNNNode From { get; set; }
        public GNNNode To { get; set; }
        public double[][] State { get; set; }
    }
}
