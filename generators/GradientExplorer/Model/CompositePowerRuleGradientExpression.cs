using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public class CompositePowerRuleGradientExpression : GradientExpression
    {
        public GradientGraph F { get; set; }
        public GradientGraph G { get; set; }
        public GradientGraph FPrime { get; set; }
        public GradientGraph GPrime { get; set; }

        public Node Differentiate()
        {
            Node baseExpression = GraphHelper.Function(NodeType.Pow, F.Nodes.FirstOrDefault(), G.Nodes.FirstOrDefault());

            Node lnF = GraphHelper.Function(NodeType.Ln, F.Nodes.FirstOrDefault());

            Node term1 = GraphHelper.Function(NodeType.Multiply, GPrime.Nodes.FirstOrDefault(), lnF);

            Node fraction = GraphHelper.Function(NodeType.Divide, FPrime.Nodes.FirstOrDefault(), F.Nodes.FirstOrDefault());

            Node term2 = GraphHelper.Function(NodeType.Multiply, G.Nodes.FirstOrDefault(), fraction);

            Node finalTerm = GraphHelper.Function(NodeType.Add, term1, term2);

            Node result = GraphHelper.Function(NodeType.Multiply, baseExpression, finalTerm);

            return result;
        }
    }
}
