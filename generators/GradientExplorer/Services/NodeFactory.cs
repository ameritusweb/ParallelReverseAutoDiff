using GradientExplorer.Model;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public class NodeFactory : INodeFactory
    {
        private INodeTypeFactory nodeTypeFactory;
        private INodeBuilderPool nodeBuilderPool;
        public NodeFactory(INodeTypeFactory nodeTypeFactory, INodeBuilderPool nodeBuilderPool)
        {
            this.nodeTypeFactory = nodeTypeFactory;
            this.nodeBuilderPool = nodeBuilderPool;
        }

        public async Task<Node> FunctionAsync(NodeType functionType, Node target)
        {
            var nodeBuilder = await nodeBuilderPool.GetNodeBuilderAsync(functionType);
            return nodeBuilder
                    .WithFunction(target)
                    .Build();
        }

        public async Task<Node> FunctionAsync(NodeType functionType, Node left, Node right)
        {
            var nodeBuilder = await nodeBuilderPool.GetNodeBuilderAsync(functionType);
            return nodeBuilder
                    .WithLeftOperand(left)
                    .WithRightOperand(right)
                    .Build();
        }

        public async Task<Node> FunctionAsync(NodeType functionType, List<Node> operands)
        {
            var nodeBuilder = await nodeBuilderPool.GetNodeBuilderAsync(functionType);
            return nodeBuilder
                    .WithOperands(operands)
                    .Build();
        }

        public async Task<Node> FunctionWithCoefficientAsync(NodeType node, Node coefficient, SyntaxNode innerInvocation)
        {
            var inner = (await ConvertToGraphAsync(innerInvocation)).Nodes.FirstOrDefault();

            var functionNode = await FunctionAsync(node, inner);

            var mult = await FunctionAsync(NodeType.Multiply, coefficient, functionNode);

            return mult;
        }

        public async Task<Node> NodeWithCoefficientAndExponentAsync(Node coefficient, Node exponent, SyntaxNode innerInvocation)
        {
            var inner = (await ConvertToGraphAsync(innerInvocation)).Nodes.FirstOrDefault();

            var pow = await NodeWithExponentAsync(inner, exponent);

            var mult = await FunctionAsync(NodeType.Multiply, coefficient, pow);

            return mult;
        }

        public async Task<Node> NodeWithExponentAsync(Node inner, Node exponent)
        {
            var nodeBuilder = await nodeBuilderPool.GetNodeBuilderAsync(NodeType.Pow);
            return nodeBuilder
                    .WithBaseOperand(inner)
                    .WithExponentOperand(exponent)
                    .Build();
        }

        public Node ToValueNode(SyntaxNode node, SyntaxToken token, LiteralType type)
        {
            var value = token.ValueText == "-" ? "-1" : token.Value;
            Node literalNode = new Node(node, node.GetType());
            literalNode.Value = value;
            literalNode.Type = type.ToString();
            return literalNode;
        }

        public Node ToConstantNode(int value)
        {
            Node literal = new Node()
            {
                Value = value,
                Type = LiteralType.Constant.ToString(),
            };
            return literal;
        }

        public Node ToConstantNode(double value)
        {
            Node literal = new Node()
            {
                Value = value,
                Type = LiteralType.Constant.ToString(),
            };
            return literal;
        }

        public Node ToValue(SyntaxNode node)
        {
            if (node is LiteralExpressionSyntax literalExpression)
            {
                Node literal = ToConstantNode((int)literalExpression.Token.Value);
                return literal;
            }
            else if (node is ElementAccessExpressionSyntax elementAccess)
            {
                Node variable = new Node()
                {
                    Value = elementAccess.Expression,
                    Type = LiteralType.Variable.ToString(),
                };
                return variable;
            }
            else
            {
                throw new InvalidOperationException("Unknown node type");
            }
        }

        public Node ToValueNodeWithParent(SyntaxNode node, Node parent, int edgeIndex)
        {
            if (node is LiteralExpressionSyntax literalExpression)
            {
                Node literal = ToValueNode(literalExpression, literalExpression.Token, LiteralType.Constant);
                parent.Edges[edgeIndex].TargetNode = literal;
            }
            else if (node is ElementAccessExpressionSyntax elementAccess)
            {
                Node variable = new Node()
                {
                    Value = elementAccess.Expression,
                    Type = LiteralType.Variable.ToString(),
                };
                parent.Edges[edgeIndex].TargetNode = variable;
            }
            else
            {
                throw new InvalidOperationException("Unknown node type");
            }

            return parent;
        }

        public async Task<GradientGraph> ConvertToGraphAsync(SyntaxNode node)
        {
            GradientGraph graph = new GradientGraph();
            if (node is InvocationExpressionSyntax invocationExpression)
            {
                var functionType = nodeTypeFactory.GetNodeType(invocationExpression);
                var args = invocationExpression.ArgumentList.Arguments;
                if (args.Count == 2)
                {
                    var innerGraph1 = await ConvertToGraphAsync(args[0].Expression);
                    var innerGraph2 = await ConvertToGraphAsync(args[1].Expression);
                    Node target1 = innerGraph1.Nodes.FirstOrDefault();
                    Node target2 = innerGraph2.Nodes.FirstOrDefault();
                    if (target1 != null && target2 != null)
                    {
                        var source = await FunctionAsync(functionType, target1, target2);
                        graph.Nodes.Add(source);
                    }
                }
                else
                {
                    var innerNode = args[0].Expression;
                    var innerGraph = await ConvertToGraphAsync(innerNode);
                    Node target = innerGraph.Nodes.FirstOrDefault();
                    if (target != null)
                    {
                        var source = await FunctionAsync(functionType, target);
                        graph.Nodes.Add(source);
                    }
                }
            }
            else if (node is LiteralExpressionSyntax literalExpression)
            {
                Node literalNode = new Node(node, node.GetType());
                literalNode.Value = literalExpression.Token.Value;
                literalNode.Type = LiteralType.Constant.ToString();
                graph.Nodes.Add(literalNode);
            }
            else if (node is ElementAccessExpressionSyntax elementAccess)
            {
                Node literalNode = new Node(node, node.GetType());
                literalNode.Value = elementAccess;
                literalNode.Type = LiteralType.Variable.ToString();
                graph.Nodes.Add(literalNode);
            }
            else if (node is ParenthesizedExpressionSyntax parenthesized)
            {
                return await ConvertToGraphAsync(parenthesized.Expression);
            }
            else if (node is PrefixUnaryExpressionSyntax prefixUnary)
            {
                Node baseNode = (await ConvertToGraphAsync(prefixUnary.Operand)).Nodes.FirstOrDefault();
                var mult = await FunctionAsync(NodeType.Multiply, ToValueNode(node, prefixUnary.OperatorToken, LiteralType.Constant), baseNode);
                graph.Nodes.Add(mult);
            }
            else if (node is BinaryExpressionSyntax binary)
            {
                Node left = (await ConvertToGraphAsync(binary.Left)).Nodes.FirstOrDefault();
                Node right = (await ConvertToGraphAsync(binary.Right)).Nodes.FirstOrDefault();
                Node multiply = await FunctionAsync(nodeTypeFactory.ToNodeType(binary), left, right);
                graph.Nodes.Add(multiply);
            }
            else
            {
                throw new InvalidOperationException("Unknown node type");
            }

            return graph;
        }
    }
}
