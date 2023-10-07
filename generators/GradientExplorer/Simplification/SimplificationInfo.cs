using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Simplification
{
    public class SimplificationInfo
    {
        public string Category { get; set; }
        public string Description { get; set; }
        public List<string> AffectedNodeIds { get; set; }
        public List<(string, string)> AffectedEdgeIds { get; set; }
    }
}
