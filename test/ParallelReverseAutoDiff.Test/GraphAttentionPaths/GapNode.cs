using ParallelReverseAutoDiff.RMAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    public class GapNode
    {
        public Guid Id { get; set; }

        public int PositionX { get; set; }

        public int PositionY { get; set; }

        public bool IsInPath { get; set; }

        public GapType GapType { get; set; }

        public Matrix FeatureVector { get; set; }

        public List<GapEdge> Edges { get; set; }
    }
}
