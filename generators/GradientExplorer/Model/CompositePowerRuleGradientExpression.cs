using GradientExplorer.Services;
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

        public Node Differentiate(INodeFactory nodeFactory)
        {
            Node baseExpression = nodeFactory.Function(NodeType.Pow, F.Nodes.FirstOrDefault(), G.Nodes.FirstOrDefault());

            Node lnF = nodeFactory.Function(NodeType.Ln, F.Nodes.FirstOrDefault());

            Node term1 = nodeFactory.Function(NodeType.Multiply, GPrime.Nodes.FirstOrDefault(), lnF);

            Node fraction = nodeFactory.Function(NodeType.Divide, FPrime.Nodes.FirstOrDefault(), F.Nodes.FirstOrDefault());

            Node term2 = nodeFactory.Function(NodeType.Multiply, G.Nodes.FirstOrDefault(), fraction);

            Node finalTerm = nodeFactory.Function(NodeType.Add, term1, term2);

            Node result = nodeFactory.Function(NodeType.Multiply, baseExpression, finalTerm);
            result.ExpressionType = GradientExpressionType.CompositePowerRule;
            return result;
        }
    }
}
