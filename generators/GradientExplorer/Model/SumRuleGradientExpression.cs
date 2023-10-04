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

        public Node Differentiate()
        {
            return new Node();
        }
    }
}
