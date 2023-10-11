using GradientExplorer.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public class LaTeXBuilder : ILaTeXBuilder
    {
        public StringBuilder GenerateLatexFromGraph(Node node, StringBuilder builder)
        {
            switch (node.NodeType)
            {
                case NodeType.Add:
                    builder.Append("\\left(");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append(" + ");
                    GenerateLatexFromGraph(node.Edges[1].TargetNode, builder);
                    builder.Append("\\right)");
                    break;
                case NodeType.Subtract:
                    builder.Append("\\left(");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append(" - ");
                    GenerateLatexFromGraph(node.Edges[1].TargetNode, builder);
                    builder.Append("\\right)");
                    break;
                case NodeType.Multiply:
                    builder.Append("\\left(");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append(" \\times ");
                    GenerateLatexFromGraph(node.Edges[1].TargetNode, builder);
                    builder.Append("\\right)");
                    break;
                case NodeType.Divide:
                    builder.Append("\\frac{");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append("}{");
                    GenerateLatexFromGraph(node.Edges[1].TargetNode, builder);
                    builder.Append("}");
                    break;
                case NodeType.Exp:
                    builder.Append("e^{");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append("}");
                    break;
                case NodeType.Sin:
                    builder.Append("\\sin\\left(");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append("\\right)");
                    break;
                case NodeType.Asin:
                    builder.Append("\\asin\\left(");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append("\\right)");
                    break;
                case NodeType.Sinh:
                    builder.Append("\\sinh\\left(");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append("\\right)");
                    break;
                case NodeType.Cos:
                    builder.Append("\\cos\\left(");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append("\\right)");
                    break;
                case NodeType.Acos:
                    builder.Append("\\acos\\left(");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append("\\right)");
                    break;
                case NodeType.Cosh:
                    builder.Append("\\cosh\\left(");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append("\\right)");
                    break;
                case NodeType.Tan:
                    builder.Append("\\tan\\left(");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append("\\right)");
                    break;
                case NodeType.Atan:
                    builder.Append("\\atan\\left(");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append("\\right)");
                    break;
                case NodeType.Tanh:
                    builder.Append("\\tanh\\left(");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append("\\right)");
                    break;
                case NodeType.Log:
                    builder.Append("\\log\\left(");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append("\\right)");
                    break;
                case NodeType.Ln:
                    builder.Append("\\ln\\left(");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append("\\right)");
                    break;
                case NodeType.Sqrt:
                    builder.Append("\\sqrt{");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append("}");
                    break;
                case NodeType.Pow:
                    // Handle base
                    builder.Append("{");
                    GenerateLatexFromGraph(node.Edges[0].TargetNode, builder);
                    builder.Append("}");

                    // Handle exponent
                    builder.Append("^{");
                    GenerateLatexFromGraph(node.Edges[1].TargetNode, builder);
                    builder.Append("}");
                    break;
                case NodeType.ConstantOrVariable:
                    builder.Append(node.ValueAsString);
                    break;
                default:
                    break;
            }
            return builder;
        }
    }
}
