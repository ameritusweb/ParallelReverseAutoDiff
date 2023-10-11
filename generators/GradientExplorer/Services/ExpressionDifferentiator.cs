using GradientExplorer.Model;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public class ExpressionDifferentiator : IExpressionDifferentiator
    {
        private INodeFactory nodeFactory;

        public ExpressionDifferentiator(INodeFactory nodeFactory)
        {
            this.nodeFactory = nodeFactory;
        }

        public Node DifferentiateLiteral(SyntaxNode node, LiteralType type)
        {
            Node literalNode = new Node(node, node.GetType());
            literalNode.Value = type == LiteralType.Constant ? 0 : 1;
            if (node is PrefixUnaryExpressionSyntax prefix)
            {
                if (prefix.OperatorToken.Text == "-")
                {
                    literalNode.Value = (int)literalNode.Value * -1;
                }
            }
            literalNode.Type = type.ToString();
            return literalNode;
        }

        public GradientGraph DifferentiateExpExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();
            var target = nodeFactory.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var node = nodeFactory.Function(NodeType.Exp, target);

            graph.Nodes.Add(node);
            return graph;
        }

        public GradientGraph DifferentiateSinExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = nodeFactory.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var node = nodeFactory.Function(NodeType.Cos, target);

            graph.Nodes.Add(node);
            return graph;
        }

        public GradientGraph DifferentiateSinhExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = nodeFactory.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var node = nodeFactory.Function(NodeType.Cosh, target);

            graph.Nodes.Add(node);
            return graph;
        }

        public GradientGraph DifferentiateAsinExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = nodeFactory.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            Node numerator = nodeFactory.ToConstantNode(1);

            Node exponent = nodeFactory.ToConstantNode(2);

            var squared = nodeFactory.NodeWithExponent(target, exponent);

            Node operand = nodeFactory.ToConstantNode(1);

            var subtract = nodeFactory.Function(NodeType.Subtract, operand, squared);

            var denominator = nodeFactory.Function(NodeType.Sqrt, subtract);

            var result = nodeFactory.Function(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(result);
            return graph;
        }

        public GradientGraph DifferentiateCosExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node coefficient = nodeFactory.ToConstantNode(-1);

            var inner = nodeFactory.FunctionWithCoefficient(NodeType.Sin, coefficient, innerInvocation);

            graph.Nodes.Add(inner);

            return graph;
        }

        public GradientGraph DifferentiateCoshExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = nodeFactory.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var inner = nodeFactory.Function(NodeType.Sinh, target);

            graph.Nodes.Add(inner);

            return graph;
        }

        public GradientGraph DifferentiateAcosExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = nodeFactory.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            Node numerator = nodeFactory.ToConstantNode(-1);

            Node exponent = nodeFactory.ToConstantNode(2);

            var squared = nodeFactory.NodeWithExponent(target, exponent);

            Node operand = nodeFactory.ToConstantNode(1);

            var subtract = nodeFactory.Function(NodeType.Subtract, operand, squared);

            var denominator = nodeFactory.Function(NodeType.Sqrt, subtract);

            var result = nodeFactory.Function(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(result);
            return graph;
        }

        public GradientGraph DifferentiateTanExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node numerator = nodeFactory.ToConstantNode(1);

            var inner = nodeFactory.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var cos = nodeFactory.Function(NodeType.Cos, inner);

            var denominator = nodeFactory.Function(NodeType.Multiply, cos, cos);

            var divide = nodeFactory.Function(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(divide);

            return graph;
        }

        public GradientGraph DifferentiateTanhExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = nodeFactory.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            Node operand = nodeFactory.ToConstantNode(1);

            var tanh = nodeFactory.Function(NodeType.Tanh, target);

            var mult = nodeFactory.Function(NodeType.Multiply, tanh, tanh);

            var result = nodeFactory.Function(NodeType.Subtract, operand, mult);

            graph.Nodes.Add(result);
            return graph;
        }

        public GradientGraph DifferentiateAtanExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = nodeFactory.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            Node numerator = nodeFactory.ToConstantNode(1);

            Node exponent = nodeFactory.ToConstantNode(2);

            var squared = nodeFactory.NodeWithExponent(target, exponent);

            Node operand = nodeFactory.ToConstantNode(1);

            var denominator = nodeFactory.Function(NodeType.Add, operand, squared);

            var result = nodeFactory.Function(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(result);
            return graph;
        }

        public GradientGraph DifferentiateLogExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node numerator = nodeFactory.ToConstantNode(1);

            var inner = nodeFactory.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            Node c = nodeFactory.ToConstantNode(10);

            var ln10 = nodeFactory.Function(NodeType.Ln, c);

            var denominator = nodeFactory.Function(NodeType.Multiply, inner, ln10);

            var divide = nodeFactory.Function(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(divide);

            return graph;
        }

        public GradientGraph DifferentiateLnExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node numerator = nodeFactory.ToConstantNode(1);

            var denominator = nodeFactory.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var divide = nodeFactory.Function(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(divide);

            return graph;
        }

        public GradientGraph DifferentiatePowExpression(List<SyntaxNode> syntaxNodes)
        {
            GradientGraph graph = new GradientGraph();

            var subtract = new Node()
            {
                NodeType = NodeType.Subtract,
            };

            var constant = nodeFactory.ToConstantNode(1);

            var edge1 = new Edge()
            {
                Relationship = RelationshipType.Operand,
            };

            var edge2 = new Edge()
            {
                Relationship = RelationshipType.Operand,
                TargetNode = constant,
            };

            subtract.Edges.Add(edge1);
            subtract.Edges.Add(edge2);

            var exponent = nodeFactory.ToValueNodeWithParent(syntaxNodes[1], subtract, 0);

            Node coefficient = nodeFactory.ToValue(syntaxNodes[1]);

            var baseNode = nodeFactory.NodeWithCoefficientAndExponent(coefficient, exponent, syntaxNodes[0]);

            graph.Nodes.Add(baseNode);

            return graph;
        }

        public GradientGraph DifferentiateSqrtExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node two = nodeFactory.ToConstantNode(2);

            var functionWithCoefficent = nodeFactory.FunctionWithCoefficient(NodeType.Sqrt, two, innerInvocation);

            Node numerator = nodeFactory.ToConstantNode(1);

            var divide = nodeFactory.Function(NodeType.Divide, numerator, functionWithCoefficent);

            graph.Nodes.Add(divide);

            return graph;
        }
    }
}
