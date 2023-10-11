using GradientExplorer.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public class SumRuleGradientExpression : GradientExpression
    {
        public List<GradientGraph> Operands { get; set; } = new List<GradientGraph>();

        public async Task<Node> DifferentiateAsync(INodeFactory nodeFactory)
        {

            var result = await nodeFactory.FunctionAsync(NodeType.Add, Operands.Select(x => x.Nodes.FirstOrDefault()).ToList());
            result.ExpressionType = GradientExpressionType.SumRule;
            return result;
        }
    }
}
