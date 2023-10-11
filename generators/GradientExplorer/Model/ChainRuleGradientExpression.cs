using GradientExplorer.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public class ChainRuleGradientExpression : GradientExpression
    {
        public GradientGraph FPrimeOfG { get; set; }
        public GradientGraph GPrime { get; set; }

        public Node Differentiate(INodeFactory nodeFactory)
        {

            var result = nodeFactory.Function(NodeType.Multiply, FPrimeOfG.Nodes.FirstOrDefault(), GPrime.Nodes.FirstOrDefault());
            result.ExpressionType = GradientExpressionType.ChainRule;
            return result;
        }
    }
}
