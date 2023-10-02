using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public class GradientGraph
    {
        public List<Node> Nodes { get; set; } = new List<Node>();

        public List<GradientExpression> Expressions { get; set; } = new List<GradientExpression>();

        public string ToLaTeX()
        {
            return "\\left( y = \\lim_{{z \\to 0}} \\frac{\\partial}{\\partial z} \\left( e^{z^2} \\right) \\right)";
        }
    }
}
