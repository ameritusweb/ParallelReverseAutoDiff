using GradientExplorer.Model;
using Microsoft.CodeAnalysis;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public interface INodeFactory
    {
        Node Function(NodeType functionType, Node target);
        Node Function(NodeType functionType, Node left, Node right);
        Node Function(NodeType functionType, List<Node> operands);
        Node FunctionWithCoefficient(NodeType nodeType, Node coefficient, SyntaxNode innerInvocation);
        Node NodeWithCoefficientAndExponent(Node coefficient, Node exponent, SyntaxNode innerInvocation);
        Node NodeWithExponent(Node inner, Node exponent);
        Node ToValueNode(SyntaxNode node, SyntaxToken token, LiteralType type);
        Node ToConstantNode(int value);
        Node ToConstantNode(double value);
        Node ToValue(SyntaxNode node);
        Node ToValueNodeWithParent(SyntaxNode node, Node parent, int edgeIndex);
        GradientGraph ConvertToGraph(SyntaxNode node);
    }
}
