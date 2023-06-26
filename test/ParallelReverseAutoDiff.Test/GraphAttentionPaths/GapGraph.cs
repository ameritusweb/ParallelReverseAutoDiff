using ParallelReverseAutoDiff.RMAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    public class GapGraph
    {
        public List<GapEdge> GapEdges { get; set; }

        public List<GapNode> GapNodes { get; set; }

        public List<GapPath> GapPaths { get; set; }

        public Matrix AdjacencyMatrix { get; set; }

        public Matrix NormalizedAdjacency { get; set; }
    }
}
