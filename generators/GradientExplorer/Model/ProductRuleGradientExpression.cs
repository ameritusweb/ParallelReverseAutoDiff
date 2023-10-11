using CSharpMath.Structures;
using GradientExplorer.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public class ProductRuleGradientExpression : GradientExpression
    {
        public GradientGraph F { get; set; }
        public GradientGraph G { get; set; }
        public GradientGraph FPrime { get; set; }
        public GradientGraph GPrime { get; set; }

        public Node Differentiate(INodeFactory nodeFactory)
        {
            Node operandLeft = nodeFactory.Function(NodeType.Multiply, FPrime.Nodes.FirstOrDefault(), G.Nodes.FirstOrDefault());

            Node operandRight = nodeFactory.Function(NodeType.Multiply, F.Nodes.FirstOrDefault(), GPrime.Nodes.FirstOrDefault());

            Node result = nodeFactory.Function(NodeType.Add, operandLeft, operandRight);
            result.ExpressionType = GradientExpressionType.ProductRule;
            return result;
        }
    }
}
