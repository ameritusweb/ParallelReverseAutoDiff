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
        Task<Node> FunctionAsync(NodeType functionType, Node target);
        Task<Node> FunctionAsync(NodeType functionType, Node left, Node right);
        Task<Node> FunctionAsync(NodeType functionType, List<Node> operands);
        Task<Node> FunctionWithCoefficientAsync(NodeType nodeType, Node coefficient, SyntaxNode innerInvocation);
        Task<Node> NodeWithCoefficientAndExponentAsync(Node coefficient, Node exponent, SyntaxNode innerInvocation);
        Task<Node> NodeWithExponentAsync(Node inner, Node exponent);
        Node ToValueNode(SyntaxNode node, SyntaxToken token, LiteralType type);
        Node ToConstantNode(int value);
        Node ToConstantNode(double value);
        Node ToValue(SyntaxNode node);
        Node ToValueNodeWithParent(SyntaxNode node, Node parent, int edgeIndex);
        Task<GradientGraph> ConvertToGraphAsync(SyntaxNode node);
    }
}
