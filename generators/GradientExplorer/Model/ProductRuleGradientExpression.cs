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

        public async Task<Node> DifferentiateAsync(INodeFactory nodeFactory)
        {
            Node operandLeft = await nodeFactory.FunctionAsync(NodeType.Multiply, FPrime.Nodes.FirstOrDefault(), G.Nodes.FirstOrDefault());

            Node operandRight = await nodeFactory.FunctionAsync(NodeType.Multiply, F.Nodes.FirstOrDefault(), GPrime.Nodes.FirstOrDefault());

            Node result = await nodeFactory.FunctionAsync(NodeType.Add, operandLeft, operandRight);
            result.ExpressionType = GradientExpressionType.ProductRule;
            return result;
        }
    }
}
