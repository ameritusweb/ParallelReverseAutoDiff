using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis;
using System.Linq;
using System.Collections.Generic;

namespace GradientExplorer.Model
{
    public static class GraphHelper
    {
        public static Node Function(NodeType functionType, Node target)
        {
            Edge edge = new Edge()
            {
                Relationship = RelationshipType.Function,
                TargetNode = target,
            };

            Node node = new Node()
            {
                NodeType = functionType,
            };
            node.Edges.Add(edge);
            return node;
        }

        public static Node Function(NodeType functionType, Node left, Node right)
        {
            Edge leftEdge = new Edge()
            {
                Relationship = RelationshipType.Operand,
                TargetNode = left,
            };

            Edge rightEdge = new Edge()
            {
                Relationship = RelationshipType.Operand,
                TargetNode = right,
            };

            Node node = new Node()
            {
                NodeType = functionType,
            };
            node.Edges.Add(leftEdge);
            node.Edges.Add(rightEdge);
            return node;
        }

        public static Node Function(NodeType functionType, List<Node> operands)
        {
            Node node = new Node()
            {
                NodeType = functionType,
            };
            
            foreach (var operand in operands)
            {
                Edge edge = new Edge()
                {
                    Relationship = RelationshipType.Operand,
                    TargetNode = operand,
                };
                node.Edges.Add(edge);
            }
            return node;
        }

        public static Node FunctionWithCoefficient(NodeType node, Node coefficient, SyntaxNode innerInvocation)
        {
            var inner = ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var functionNode = Function(node, inner);

            var mult = Function(NodeType.Multiply, coefficient, functionNode);

            return mult;
        }

        public static Node NodeWithCoefficientAndExponent(Node coefficient, Node exponent, SyntaxNode innerInvocation)
        {
            var inner = ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var pow = NodeWithExponent(inner, exponent);

            var mult = Function(NodeType.Multiply, coefficient, pow);

            return mult;
        }

        public static Node NodeWithExponent(Node inner, Node exponent)
        {
            Node pow = new Node()
            {
                NodeType = NodeType.Pow,
            };

            Edge baseEdge = new Edge()
            {
                Relationship = RelationshipType.Operand,
                TargetNode = inner,
            };

            Edge exponentEdge = new Edge()
            {
                Relationship = RelationshipType.Operand,
                TargetNode = exponent,
            };
            pow.Edges.Add(baseEdge);
            pow.Edges.Add(exponentEdge);

            return pow;
        }

        public static Node ToValueNode(SyntaxNode node, SyntaxToken token, LiteralType type)
        {
            Node literalNode = new Node(node, node.GetType());
            literalNode.Value = token.Value;
            literalNode.Type = type.ToString();
            return literalNode;
        }

        public static GradientGraph ConvertToGraph(SyntaxNode node)
        {
            GradientGraph graph = new GradientGraph();
            if (node is InvocationExpressionSyntax invocationExpression)
            {
                var functionType = GetNodeType(invocationExpression);
                var innerNode = invocationExpression.ArgumentList.Arguments[0].Expression;
                var innerGraph = ConvertToGraph(innerNode);
                Node target = innerGraph.Nodes.FirstOrDefault();
                if (target != null)
                {
                    var source = Function(functionType, target);
                    graph.Nodes.Add(source);
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
                literalNode.Value = elementAccess.Expression;
                literalNode.Type = LiteralType.Variable.ToString();
                graph.Nodes.Add(literalNode);
            }
            else if (node is ParenthesizedExpressionSyntax parenthesized)
            {
                return ConvertToGraph(parenthesized.Expression);
            }
            else if (node is PrefixUnaryExpressionSyntax prefixUnary)
            {
                Node baseNode = ConvertToGraph(prefixUnary.Operand).Nodes.FirstOrDefault();
                var mult = Function(NodeType.Multiply, baseNode, ToValueNode(node, prefixUnary.OperatorToken, LiteralType.Constant));
                graph.Nodes.Add(mult);
            }
            else if (node is BinaryExpressionSyntax binary)
            {
                Node left = ConvertToGraph(binary.Left).Nodes.FirstOrDefault();
                Node right = ConvertToGraph(binary.Right).Nodes.FirstOrDefault();
                Node multiply = Function(NodeType.Multiply, left, right);
                graph.Nodes.Add(multiply);
            }
            else
            {
                throw new InvalidOperationException("Unknown node type");
            }

            return graph;
        }

        public static NodeType GetNodeType(InvocationExpressionSyntax invocation)
        {
            if (invocation.Expression is MemberAccessExpressionSyntax memberAccessExpression)
            {
                if (memberAccessExpression.Expression is IdentifierNameSyntax identifierName)
                {
                    string name = identifierName.Identifier.Text + "." + memberAccessExpression.Name.Identifier.Text;
                    switch (name)
                    {
                        case "Math.Exp":
                            return NodeType.Exp;
                        case "Math.Log":
                            return NodeType.Ln;
                        case "Math.Log10":
                            return NodeType.Log;
                        case "Math.Sin":
                            return NodeType.Sin;
                        case "Math.Asin":
                            return NodeType.Asin;
                        case "Math.Sinh":
                            return NodeType.Sinh;
                        case "Math.Cos":
                            return NodeType.Cos;
                        case "Math.Acos":
                            return NodeType.Acos;
                        case "Math.Cosh":
                            return NodeType.Cosh;
                        case "Math.Tan":
                            return NodeType.Tan;
                        case "Math.Atan":
                            return NodeType.Atan;
                        case "Math.Tanh":
                            return NodeType.Tanh;
                        case "Math.Sqrt":
                            return NodeType.Sqrt;
                        case "Math.Pow":
                            return NodeType.Pow;
                        default:
                            throw new InvalidOperationException("Unknown function type");
                    }
                }
            }

            return NodeType.ConstantOrVariable;
        }
    }
}
