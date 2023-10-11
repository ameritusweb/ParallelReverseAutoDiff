using GradientExplorer.Services;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GradientExplorer.Model
{
    public class GradientGraph
    {
        public List<Node> Nodes { get; set; } = new List<Node>();

        public List<GradientExpression> Expressions { get; set; } = new List<GradientExpression>();

        public GradientGraph DeepCopy()
        {
            GradientGraph result = new GradientGraph();
            foreach (var node in Nodes)
            {
                result.Nodes.Add(node.DeepCopy());
            }
            return result;
        }

        public string ToLaTeX(ILaTeXBuilder laTeXBuilder)
        {
            var node = Nodes.FirstOrDefault();
            StringBuilder builder = new StringBuilder();
            builder = laTeXBuilder.GenerateLatexFromGraph(node, builder);
            return builder.ToString();
        }
    }
}
