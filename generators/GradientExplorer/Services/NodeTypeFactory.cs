using GradientExplorer.Model;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public class NodeTypeFactory
    {
        public NodeType ToNodeType(BinaryExpressionSyntax binaryExpression)
        {
            if (binaryExpression.OperatorToken.Text == "+")
            {
                return NodeType.Add;
            }
            else if (binaryExpression.OperatorToken.Text == "-")
            {
                return NodeType.Subtract;
            }
            else if (binaryExpression.OperatorToken.Text == "*")
            {
                return NodeType.Multiply;
            }
            else if (binaryExpression.OperatorToken.Text == "/")
            {
                return NodeType.Divide;
            }
            else
            {
                throw new InvalidOperationException("Unknown operator");
            }
        }

        public NodeType GetNodeType(InvocationExpressionSyntax invocation)
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
