using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public class Edge
    {

        public Edge DeepCopy()
        {
            Edge copy = new Edge();
            copy.Relationship = Relationship;
            copy.TargetNode = TargetNode.DeepCopy();
            return copy;
        }

        public RelationshipType Relationship { get; set; } // "operand of", "exponent of", etc.
        public Node TargetNode { get; set; }
    }
}
