using ParallelReverseAutoDiff.RMAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    public class GapEdge
    {
        public GapNode Node { get; set; }

        public Matrix FeatureVector { get; set; }
    }
}
