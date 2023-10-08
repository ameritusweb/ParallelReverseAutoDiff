using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis;
using System.Collections.Generic;
using System.Linq;

namespace GradientExplorer.Model
{
    public static class DifferentiationHelper
    {
        public static Node DifferentiateLiteral(SyntaxNode node, LiteralType type)
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

        public static GradientGraph DifferentiateExpExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();
            var target = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var node = GraphHelper.Function(NodeType.Exp, target);

            graph.Nodes.Add(node);
            return graph;
        }

        public static GradientGraph DifferentiateSinExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var node = GraphHelper.Function(NodeType.Cos, target);

            graph.Nodes.Add(node);
            return graph;
        }

        public static GradientGraph DifferentiateSinhExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var node = GraphHelper.Function(NodeType.Cosh, target);

            graph.Nodes.Add(node);
            return graph;
        }

        public static GradientGraph DifferentiateAsinExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            Node numerator = GraphHelper.ToConstantNode(1);

            Node exponent = GraphHelper.ToConstantNode(2);

            var squared = GraphHelper.NodeWithExponent(target, exponent);

            Node operand = GraphHelper.ToConstantNode(1);

            var subtract = GraphHelper.Function(NodeType.Subtract, operand, squared);

            var denominator = GraphHelper.Function(NodeType.Sqrt, subtract);

            var result = GraphHelper.Function(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(result);
            return graph;
        }

        public static GradientGraph DifferentiateCosExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node coefficient = GraphHelper.ToConstantNode(-1);

            var inner = GraphHelper.FunctionWithCoefficient(NodeType.Sin, coefficient, innerInvocation);

            graph.Nodes.Add(inner);

            return graph;
        }

        public static GradientGraph DifferentiateCoshExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var inner = GraphHelper.Function(NodeType.Sinh, target);

            graph.Nodes.Add(inner);

            return graph;
        }

        public static GradientGraph DifferentiateAcosExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            Node numerator = GraphHelper.ToConstantNode(-1);

            Node exponent = GraphHelper.ToConstantNode(2);

            var squared = GraphHelper.NodeWithExponent(target, exponent);

            Node operand = GraphHelper.ToConstantNode(1);

            var subtract = GraphHelper.Function(NodeType.Subtract, operand, squared);

            var denominator = GraphHelper.Function(NodeType.Sqrt, subtract);

            var result = GraphHelper.Function(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(result);
            return graph;
        }

        public static GradientGraph DifferentiateTanExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node numerator = GraphHelper.ToConstantNode(1);

            var inner = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var cos = GraphHelper.Function(NodeType.Cos, inner);

            var denominator = GraphHelper.Function(NodeType.Multiply, cos, cos);

            var divide = GraphHelper.Function(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(divide);

            return graph;
        }

        public static GradientGraph DifferentiateTanhExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            Node operand = GraphHelper.ToConstantNode(1);

            var tanh = GraphHelper.Function(NodeType.Tanh, target);

            var mult = GraphHelper.Function(NodeType.Multiply, tanh, tanh);

            var result = GraphHelper.Function(NodeType.Subtract, operand, mult);

            graph.Nodes.Add(result);
            return graph;
        }

        public static GradientGraph DifferentiateAtanExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            Node numerator = GraphHelper.ToConstantNode(1);

            Node exponent = GraphHelper.ToConstantNode(2);

            var squared = GraphHelper.NodeWithExponent(target, exponent);

            Node operand = GraphHelper.ToConstantNode(1);

            var denominator = GraphHelper.Function(NodeType.Add, operand, squared);

            var result = GraphHelper.Function(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(result);
            return graph;
        }

        public static GradientGraph DifferentiateLogExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node numerator = GraphHelper.ToConstantNode(1);

            var inner = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            Node c = GraphHelper.ToConstantNode(10);

            var ln10 = GraphHelper.Function(NodeType.Ln, c);

            var denominator = GraphHelper.Function(NodeType.Multiply, inner, ln10);

            var divide = GraphHelper.Function(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(divide);

            return graph;
        }

        public static GradientGraph DifferentiateLnExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node numerator = GraphHelper.ToConstantNode(1);

            var denominator = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var divide = GraphHelper.Function(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(divide);

            return graph;
        }

        public static GradientGraph DifferentiatePowExpression(List<SyntaxNode> syntaxNodes)
        {
            GradientGraph graph = new GradientGraph();

            var subtract = new Node()
            {
                NodeType = NodeType.Subtract,
            };

            var constant = GraphHelper.ToConstantNode(1);

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

            var exponent = GraphHelper.ToValueNodeWithParent(syntaxNodes[1], subtract, 0);

            Node coefficient = GraphHelper.ToValue(syntaxNodes[1]);

            var baseNode = GraphHelper.NodeWithCoefficientAndExponent(coefficient, exponent, syntaxNodes[0]);

            graph.Nodes.Add(baseNode);

            return graph;
        }

        public static GradientGraph DifferentiateSqrtExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node two = GraphHelper.ToConstantNode(2);

            var functionWithCoefficent = GraphHelper.FunctionWithCoefficient(NodeType.Sqrt, two, innerInvocation);

            Node numerator = GraphHelper.ToConstantNode(1);

            var divide = GraphHelper.Function(NodeType.Divide, numerator, functionWithCoefficent);

            graph.Nodes.Add(divide);

            return graph;
        }
    }
}
