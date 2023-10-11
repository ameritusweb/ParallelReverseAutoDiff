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

        public async Task<Node> DifferentiateAsync(INodeFactory nodeFactory)
        {
            Node baseExpression = await nodeFactory.FunctionAsync(NodeType.Pow, F.Nodes.FirstOrDefault(), G.Nodes.FirstOrDefault());

            Node lnF = await nodeFactory.FunctionAsync(NodeType.Ln, F.Nodes.FirstOrDefault());

            Node term1 = await nodeFactory.FunctionAsync(NodeType.Multiply, GPrime.Nodes.FirstOrDefault(), lnF);

            Node fraction = await nodeFactory.FunctionAsync(NodeType.Divide, FPrime.Nodes.FirstOrDefault(), F.Nodes.FirstOrDefault());

            Node term2 = await nodeFactory.FunctionAsync(NodeType.Multiply, G.Nodes.FirstOrDefault(), fraction);

            Node finalTerm = await nodeFactory.FunctionAsync(NodeType.Add, term1, term2);

            Node result = await nodeFactory.FunctionAsync(NodeType.Multiply, baseExpression, finalTerm);
            result.ExpressionType = GradientExpressionType.CompositePowerRule;
            return result;
        }
    }
}
