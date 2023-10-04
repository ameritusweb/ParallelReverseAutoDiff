using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public class UnaryGradientExpression : GradientExpression
    {
        public GradientGraph FPrime { get; set; }

        public Node Differentiate()
        {
            return FPrime.Nodes.FirstOrDefault();
        }
    }
}
