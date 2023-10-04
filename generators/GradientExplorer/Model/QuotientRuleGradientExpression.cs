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
            return new Node();
        }
    }
}
