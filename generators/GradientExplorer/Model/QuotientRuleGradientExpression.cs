using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public class QuotientRuleGradientExpression : GradientExpression
    {
        public GradientGraph F { get; set; }
        public GradientGraph G { get; set; }
        public GradientGraph FPrime { get; set; }
        public GradientGraph GPrime { get; set; }

        public Node Differentiate()
        {

            Node operandLeft = GraphHelper.Function(NodeType.Multiply, FPrime.Nodes.FirstOrDefault(), G.Nodes.FirstOrDefault());

            Node operandRight = GraphHelper.Function(NodeType.Multiply, F.Nodes.FirstOrDefault(), GPrime.Nodes.FirstOrDefault());

            Node numerator = GraphHelper.Function(NodeType.Subtract, operandLeft, operandRight);

            Node denominator = GraphHelper.NodeWithExponent(G.Nodes.FirstOrDefault(), new Node() { Value = 2, Type = LiteralType.Constant.ToString() });

            Node result = GraphHelper.Function(NodeType.Divide, numerator, denominator);
            result.ExpressionType = GradientExpressionType.QuotientRule;
            return result;
        }
    }
}
