using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public class Edge
    {
        public RelationshipType Relationship { get; set; } // "operand of", "exponent of", etc.
        public Node TargetNode { get; set; }
    }
}
