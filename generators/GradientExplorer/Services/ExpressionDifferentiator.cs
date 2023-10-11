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

        public async Task<GradientGraph> DifferentiateExpExpressionAsync(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();
            var target = (await nodeFactory.ConvertToGraphAsync(innerInvocation)).Nodes.FirstOrDefault();

            var node = await nodeFactory.FunctionAsync(NodeType.Exp, target);

            graph.Nodes.Add(node);
            return graph;
        }

        public async Task<GradientGraph> DifferentiateSinExpressionAsync(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = (await nodeFactory.ConvertToGraphAsync(innerInvocation)).Nodes.FirstOrDefault();

            var node = await nodeFactory.FunctionAsync(NodeType.Cos, target);

            graph.Nodes.Add(node);
            return graph;
        }

        public async Task<GradientGraph> DifferentiateSinhExpressionAsync(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = (await nodeFactory.ConvertToGraphAsync(innerInvocation)).Nodes.FirstOrDefault();

            var node = await nodeFactory.FunctionAsync(NodeType.Cosh, target);

            graph.Nodes.Add(node);
            return graph;
        }

        public async Task<GradientGraph> DifferentiateAsinExpressionAsync(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = (await nodeFactory.ConvertToGraphAsync(innerInvocation)).Nodes.FirstOrDefault();

            Node numerator = nodeFactory.ToConstantNode(1);

            Node exponent = nodeFactory.ToConstantNode(2);

            var squared = await nodeFactory.NodeWithExponentAsync(target, exponent);

            Node operand = nodeFactory.ToConstantNode(1);

            var subtract = await nodeFactory.FunctionAsync(NodeType.Subtract, operand, squared);

            var denominator = await nodeFactory.FunctionAsync(NodeType.Sqrt, subtract);

            var result = await nodeFactory.FunctionAsync(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(result);
            return graph;
        }

        public async Task<GradientGraph> DifferentiateCosExpressionAsync(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node coefficient = nodeFactory.ToConstantNode(-1);

            var inner = await nodeFactory.FunctionWithCoefficientAsync(NodeType.Sin, coefficient, innerInvocation);

            graph.Nodes.Add(inner);

            return graph;
        }

        public async Task<GradientGraph> DifferentiateCoshExpressionAsync(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = (await nodeFactory.ConvertToGraphAsync(innerInvocation)).Nodes.FirstOrDefault();

            var inner = await nodeFactory.FunctionAsync(NodeType.Sinh, target);

            graph.Nodes.Add(inner);

            return graph;
        }

        public async Task<GradientGraph> DifferentiateAcosExpressionAsync(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = (await nodeFactory.ConvertToGraphAsync(innerInvocation)).Nodes.FirstOrDefault();

            Node numerator = nodeFactory.ToConstantNode(-1);

            Node exponent = nodeFactory.ToConstantNode(2);

            var squared = await nodeFactory.NodeWithExponentAsync(target, exponent);

            Node operand = nodeFactory.ToConstantNode(1);

            var subtract = await nodeFactory.FunctionAsync(NodeType.Subtract, operand, squared);

            var denominator = await nodeFactory.FunctionAsync(NodeType.Sqrt, subtract);

            var result = await nodeFactory.FunctionAsync(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(result);
            return graph;
        }

        public async Task<GradientGraph> DifferentiateTanExpressionAsync(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node numerator = nodeFactory.ToConstantNode(1);

            var inner = (await nodeFactory.ConvertToGraphAsync(innerInvocation)).Nodes.FirstOrDefault();

            var cos = await nodeFactory.FunctionAsync(NodeType.Cos, inner);

            var denominator = await nodeFactory.FunctionAsync(NodeType.Multiply, cos, cos);

            var divide = await nodeFactory.FunctionAsync(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(divide);

            return graph;
        }

        public async Task<GradientGraph> DifferentiateTanhExpressionAsync(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = (await nodeFactory.ConvertToGraphAsync(innerInvocation)).Nodes.FirstOrDefault();

            Node operand = nodeFactory.ToConstantNode(1);

            var tanh = await nodeFactory.FunctionAsync(NodeType.Tanh, target);

            var mult = await nodeFactory.FunctionAsync(NodeType.Multiply, tanh, tanh);

            var result = await nodeFactory.FunctionAsync(NodeType.Subtract, operand, mult);

            graph.Nodes.Add(result);
            return graph;
        }

        public async Task<GradientGraph> DifferentiateAtanExpressionAsync(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = (await nodeFactory.ConvertToGraphAsync(innerInvocation)).Nodes.FirstOrDefault();

            Node numerator = nodeFactory.ToConstantNode(1);

            Node exponent = nodeFactory.ToConstantNode(2);

            var squared = await nodeFactory.NodeWithExponentAsync(target, exponent);

            Node operand = nodeFactory.ToConstantNode(1);

            var denominator = await nodeFactory.FunctionAsync(NodeType.Add, operand, squared);

            var result = await nodeFactory.FunctionAsync(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(result);
            return graph;
        }

        public async Task<GradientGraph> DifferentiateLogExpressionAsync(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node numerator = nodeFactory.ToConstantNode(1);

            var inner = (await nodeFactory.ConvertToGraphAsync(innerInvocation)).Nodes.FirstOrDefault();

            Node c = nodeFactory.ToConstantNode(10);

            var ln10 = await nodeFactory.FunctionAsync(NodeType.Ln, c);

            var denominator = await nodeFactory.FunctionAsync(NodeType.Multiply, inner, ln10);

            var divide = await nodeFactory.FunctionAsync(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(divide);

            return graph;
        }

        public async Task<GradientGraph> DifferentiateLnExpressionAsync(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node numerator = nodeFactory.ToConstantNode(1);

            var denominator = (await nodeFactory.ConvertToGraphAsync(innerInvocation)).Nodes.FirstOrDefault();

            var divide = await nodeFactory.FunctionAsync(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(divide);

            return graph;
        }

        public async Task<GradientGraph> DifferentiatePowExpressionAsync(List<SyntaxNode> syntaxNodes)
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

            var baseNode = await nodeFactory.NodeWithCoefficientAndExponentAsync(coefficient, exponent, syntaxNodes[0]);

            graph.Nodes.Add(baseNode);

            return graph;
        }

        public async Task<GradientGraph> DifferentiateSqrtExpressionAsync(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node two = nodeFactory.ToConstantNode(2);

            var functionWithCoefficent = await nodeFactory.FunctionWithCoefficientAsync(NodeType.Sqrt, two, innerInvocation);

            Node numerator = nodeFactory.ToConstantNode(1);

            var divide = await nodeFactory.FunctionAsync(NodeType.Divide, numerator, functionWithCoefficent);

            graph.Nodes.Add(divide);

            return graph;
        }
    }
}
