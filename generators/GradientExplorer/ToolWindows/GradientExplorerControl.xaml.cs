using CSharpMath.Rendering.FrontEnd;
using GradientExplorer.LaTeX.Wpf;
using GradientExplorer.Model;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace ToolWindow
{
    public partial class GradientExplorerControl : UserControl
    {
        private Dictionary<NodeType, Func<SyntaxNode, GradientGraph>> gradientUnaryExpressionMap;
        private Dictionary<NodeType, Func<List<SyntaxNode>, GradientGraph>> gradientNonUnaryExpressionMap;

        public GradientExplorerControl(Version vsVersion)
        {
            InitializeComponent();

            gradientUnaryExpressionMap = new Dictionary<NodeType, Func<SyntaxNode, GradientGraph>>()
            {
                { NodeType.Exp, DifferentiateExpExpression },
                { NodeType.Sin, DifferentiateSinExpression },
                { NodeType.Cos, DifferentiateCosExpression },
                { NodeType.Tan, DifferentiateTanExpression },
                { NodeType.Log, DifferentiateLogExpression },
                { NodeType.Ln, DifferentiateLnExpression },
                { NodeType.Sqrt, DifferentiateSqrtExpression },
            };

            gradientNonUnaryExpressionMap = new Dictionary<NodeType, Func<List<SyntaxNode>, GradientGraph>>()
            {
                { NodeType.Pow, DifferentiatePowExpression },
                // Add other non-unary functions here
            };

            lblHeadline.Content = $"Visual Studio v{vsVersion}";
        }

        /// <summary>
        /// Finds the gradient of the current forward function and displays it in the tool window.
        /// Similar to:
        /// public Matrix Forward(Matrix input)
        /// {
        ///     this.input = input;
        ///     int numRows = input.Length;
        ///     int numCols = input[0].Length;
        ///
        ///     this.Output = new Matrix(numRows, numCols);
        ///     for (int i = 0; i < numRows; i++)
        ///     {
        ///         for (int j = 0; j < numCols; j++)
        ///         {
        ///             this.Output[i][j] = 1.0 / (1.0 + Math.Exp(Math.Sin(-input[i][j])));
        ///         }
        ///     }
        ///
        ///     return this.Output;
        /// }
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="e">The event arge.</param>
        private async void button1_Click(object sender, RoutedEventArgs e)
        {
            var docView = await VS.Documents.GetActiveDocumentViewAsync();
            if (docView != null)
            {
                var snapshot = docView.TextView.TextSnapshot;
                var text = string.Join("\n", snapshot.Lines.Select(x => x.GetText()));

                // Convert the text into a SourceText object for Roslyn to understand
                var sourceText = SourceText.From(text);

                // Parse the SourceText into a SyntaxTree
                var syntaxTree = CSharpSyntaxTree.ParseText(sourceText);

                var root = syntaxTree.GetRoot();
                var methods = root.DescendantNodes().OfType<MethodDeclarationSyntax>();
                var forwardMethod = methods.FirstOrDefault(m => m.Identifier.Text == "Forward");
                this.ParseMethod(forwardMethod);
                var canvas = laTeXCanvas;
                WpfMathPainter painter = new WpfMathPainter();
                painter.LaTeX = "\\frac{1}{1 + e^{\\sin(-x)}}";
                painter.Draw(canvas);
            }
        }

        private void ParseMethod(MethodDeclarationSyntax method)
        {
            if (method != null)
            {
                var blockSyntax = method.Body;
                var forLoops = blockSyntax.DescendantNodes().OfType<ForStatementSyntax>().ToList();

                // Assume the last for-loop is the innermost loop
                if (forLoops.Count > 0)
                {
                    var innermostLoop = forLoops.Last();
                    var innerBlock = innermostLoop.Statement as BlockSyntax;

                    if (innerBlock != null)
                    {
                        foreach (var statement in innerBlock.Statements)
                        {
                            // Assuming that the main computation will be an expression statement.
                            if (statement is ExpressionStatementSyntax expressionStatement)
                            {
                                // This is where you'll find operations like Output[i][j] = ...
                                var expression = expressionStatement.Expression;

                                if (expression is AssignmentExpressionSyntax assignmentExpression)
                                {
                                    // This is where you'll find the right-hand side of the assignment
                                    var rightHandSide = assignmentExpression.Right;

                                    // Decompose the right-hand side into a gradient graph
                                    GradientGraph gradientGraph = DecomposeExpression(rightHandSide, new GradientGraph());
                                }
                            }
                        }
                    }
                }
            }
        }

        private GradientGraph DecomposeExpression(ExpressionSyntax expression, GradientGraph gradientGraph)
        {
            switch (expression)
            {
                case ParenthesizedExpressionSyntax parenthesizedExpression:
                    // Recursively decompose the inner expression
                    gradientGraph = DecomposeExpression(parenthesizedExpression.Expression, gradientGraph);
                    break;
                case BinaryExpressionSyntax binaryExpression:
                    // Determine the rule based on the operator
                    if (binaryExpression.OperatorToken.Text == "+")
                    {
                        // Sum rule
                        SumRuleGradientExpression gradientExpression = new SumRuleGradientExpression();
                        // Recursively decompose left and right operands
                        gradientExpression.Operands.Add(DecomposeExpression(binaryExpression.Left, gradientGraph));
                        gradientExpression.Operands.Add(DecomposeExpression(binaryExpression.Right, gradientGraph));
                        gradientGraph.Expressions.Add(gradientExpression);
                    }
                    else if (binaryExpression.OperatorToken.Text == "*")
                    {
                        // Product rule
                        ProductRuleGradientExpression gradientExpression = new ProductRuleGradientExpression();
                        // Recursively decompose left and right operands
                        gradientExpression.F = DecomposeExpression(binaryExpression.Left, gradientGraph);
                        gradientExpression.G = DecomposeExpression(binaryExpression.Right, gradientGraph);
                        gradientExpression.FPrime = Differentiate(gradientExpression.F);
                        gradientExpression.GPrime = Differentiate(gradientExpression.G);
                        gradientGraph.Expressions.Add(gradientExpression);
                    }
                    else if (binaryExpression.OperatorToken.Text == "/")
                    {
                        // Quotient rule
                        QuotientRuleGradientExpression gradientExpression = new QuotientRuleGradientExpression();
                        // Recursively decompose left and right operands
                        gradientExpression.F = DecomposeExpression(binaryExpression.Left, gradientGraph);
                        gradientExpression.G = DecomposeExpression(binaryExpression.Right, gradientGraph);
                        gradientExpression.FPrime = Differentiate(gradientExpression.F);
                        gradientExpression.GPrime = Differentiate(gradientExpression.G);
                        gradientGraph.Expressions.Add(gradientExpression);
                    }

                    break;

                case InvocationExpressionSyntax invocationExpression:
                    // Handle functions like Math.Exp, etc.
                    var functionType = this.GetNodeType(invocationExpression);
                    if (gradientNonUnaryExpressionMap.ContainsKey(functionType)) // Non-unary functions
                    {
                        var differentiationFunction = gradientNonUnaryExpressionMap[functionType];
                        var innerExpression = invocationExpression.ArgumentList.Arguments[0].Expression;
                        var secondInnerExpression = invocationExpression.ArgumentList.Arguments[1].Expression;
                        if (innerExpression is InvocationExpressionSyntax innerInvocationExpression)
                        {
                            if (secondInnerExpression is InvocationExpressionSyntax secondInnerInvocationExpression)
                            {
                                // Handle nested functions like Math.Pow(Math.Cos(x), Math.Sin(x))
                                var innerNodeType = this.GetNodeType(innerInvocationExpression);
                                var innerDifferentiationFunction = gradientUnaryExpressionMap[innerNodeType];
                                var innerInnerExpression = innerInvocationExpression.ArgumentList.Arguments[0].Expression;
                                var secondInnerNodeType = this.GetNodeType(secondInnerInvocationExpression);
                                var secondInnerDifferentiationFunction = gradientUnaryExpressionMap[secondInnerNodeType];
                                var secondInnerInnerExpression = secondInnerInvocationExpression.ArgumentList.Arguments[0].Expression;
                                CompositePowerRuleGradientExpression gradientExpression = new CompositePowerRuleGradientExpression();
                                gradientExpression.F = DecomposeExpression(innerInnerExpression, gradientGraph);
                                gradientExpression.G = DecomposeExpression(secondInnerInvocationExpression, gradientGraph);
                                gradientExpression.FPrime = innerDifferentiationFunction.Invoke(innerInnerExpression);
                                gradientExpression.GPrime = secondInnerDifferentiationFunction.Invoke(secondInnerInnerExpression);
                                gradientGraph.Expressions.Add(gradientExpression);
                            }
                        }
                        else
                        {
                            // Handle functions like Math.Pow(x, 2)
                            UnaryGradientExpression gradientExpression = new UnaryGradientExpression();
                            gradientExpression.FPrime = differentiationFunction.Invoke(new List<SyntaxNode>() { innerExpression, secondInnerExpression });
                            gradientGraph.Expressions.Add(gradientExpression);
                        }
                    }
                    else if (gradientUnaryExpressionMap.ContainsKey(functionType))
                    {
                        var differentiationFunction = gradientUnaryExpressionMap[functionType];
                        var innerExpression = invocationExpression.ArgumentList.Arguments[0].Expression;
                        if (innerExpression is InvocationExpressionSyntax innerInvocationExpression)
                        {
                            // Handle nested functions like Math.Sin(Math.Cos(x))
                            var innerNodeType = this.GetNodeType(innerInvocationExpression);
                            var innerDifferentiationFunction = gradientUnaryExpressionMap[innerNodeType];
                            var innerInnerExpression = innerInvocationExpression.ArgumentList.Arguments[0].Expression;
                            ChainRuleGradientExpression gradientExpression = new ChainRuleGradientExpression();
                            gradientExpression.FPrimeOfG = differentiationFunction.Invoke(innerInvocationExpression);
                            gradientExpression.GPrime = innerDifferentiationFunction.Invoke(innerInnerExpression);
                            gradientGraph.Expressions.Add(gradientExpression);
                        }
                        else
                        {
                            // Handle functions like Math.Exp(x)
                            UnaryGradientExpression gradientExpression = new UnaryGradientExpression();
                            gradientExpression.FPrime = differentiationFunction.Invoke(invocationExpression);
                            gradientGraph.Expressions.Add(gradientExpression);
                        }
                    }
                    break;

                // ... handle other types of expressions

                // Base cases: expression cannot be decomposed further (e.g., literals, identifiers)
                case LiteralExpressionSyntax literalExpression:
                    Node literalNode = new Node(literalExpression.Token.Value, literalExpression.Token.Value.GetType());
                    gradientGraph.Nodes.Add(literalNode);
                    break;

                default:
                    break;
            }

            return gradientGraph;
        }

        private NodeType GetNodeType(InvocationExpressionSyntax invocation)
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
                        case "Math.Cos":
                            return NodeType.Cos;
                        case "Math.Tan":
                            return NodeType.Tan;
                        case "Math.Sqrt":
                            return NodeType.Sqrt;
                        case "Math.Pow":
                            return NodeType.Pow;
                        default:
                            throw new InvalidOperationException("Unknown function type");
                    }
                }
            }

            return NodeType.Unknown;
        }

        private GradientGraph DifferentiateExpExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();
            if (innerInvocation is InvocationExpressionSyntax invocation)
            {
                Node target = new Node()
                {
                    SyntaxNode = invocation,
                };
                graph.Nodes.Add(target);

                Edge edge = new Edge()
                {
                    Relationship = RelationshipType.Function,
                    TargetNode = target,
                };

                Node node = new Node()
                {
                    NodeType = NodeType.Exp,
                };
                node.Edges.Add(edge);

                graph.Nodes.Add(node);
            }
            return graph;
        }

        private GradientGraph DifferentiateSinExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();
            if (innerInvocation is PrefixUnaryExpressionSyntax prefixUnary)
            {
                Node target = new Node()
                {
                    SyntaxNode = prefixUnary,
                };
                graph.Nodes.Add(target);

                Edge edge = new Edge()
                {
                    Relationship = RelationshipType.Function,
                    TargetNode = target,
                };

                Node node = new Node()
                {
                    NodeType = NodeType.Cos,
                };
                node.Edges.Add(edge);

                graph.Nodes.Add(node);
            }
            return graph;
        }

        private GradientGraph DifferentiateCosExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            return graph;
        }

        private GradientGraph DifferentiateTanExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            return graph;
        }

        private GradientGraph DifferentiateLogExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            return graph;
        }

        private GradientGraph DifferentiateLnExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node divide = new Node()
            {
                NodeType = NodeType.Divide,
            };

            Node numerator = new Node()
            {
                Value = 1,
                Type = typeof(int).Name,
            };

            Node denominator = new Node()
            {
                SyntaxNode = innerInvocation,
            };

            Edge operand1 = new Edge()
            {
                Relationship = RelationshipType.Numerator,
                TargetNode = numerator,
            };

            Edge operand2 = new Edge()
            {
                Relationship = RelationshipType.Denominator,
                TargetNode = denominator,
            };

            divide.Edges.Add(operand1);
            divide.Edges.Add(operand2);

            graph.Nodes.Add(divide);

            return graph;
        }

        private Node FunctionWithCoefficient(NodeType node, Node coefficient, SyntaxNode innerInvocation)
        {
            Node inner = new Node()
            {
                SyntaxNode = innerInvocation,
            };

            Edge edge = new Edge()
            {
                Relationship = RelationshipType.Function,
                TargetNode = inner,
            };

            Node functionNode = new Node()
            {
                NodeType = node,
            };
            functionNode.Edges.Add(edge);

            Edge coefficientEdge = new Edge()
            {
                Relationship = RelationshipType.Coefficient,
                TargetNode = coefficient,
            };
            functionNode.Edges.Add(coefficientEdge);

            return functionNode;
        }

        private GradientGraph DifferentiatePowExpression(List<SyntaxNode> syntaxNodes)
        {
            GradientGraph graph = new GradientGraph();

            Node baseNode = new Node()
            {
                SyntaxNode = syntaxNodes[0]
            };

            Node exponent = new Node()
            {
                Value = int.Parse((syntaxNodes[1] as LiteralExpressionSyntax).Token.Value.ToString()) - 1,
                Type = typeof(int).Name,
            };

            Edge edge = new Edge()
            {
                Relationship = RelationshipType.Exponent,
                TargetNode = exponent,
            };
            baseNode.Edges.Add(edge);

            Node coefficient = new Node()
            {
                Value = int.Parse((syntaxNodes[1] as LiteralExpressionSyntax).Token.Value.ToString()),
                Type = typeof(int).Name,
            };

            Edge edge1 = new Edge()
            {
                Relationship = RelationshipType.Coefficient,
                TargetNode = coefficient,
            };
            baseNode.Edges.Add(edge1);

            graph.Nodes.Add(baseNode);

            return graph;
        }

        private GradientGraph DifferentiateSqrtExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node two = new Node()
            {
                Value = 2,
                Type = typeof(int).Name,
            };

            var functionWithCoefficent = FunctionWithCoefficient(NodeType.Sqrt, two, innerInvocation);

            Node divide = new Node()
            {
                NodeType = NodeType.Divide
            };

            Node numerator = new Node()
            {
                Value = 1,
                Type = typeof(int).Name,
            };

            Edge operand1 = new Edge()
            {
                Relationship = RelationshipType.Numerator,
                TargetNode = numerator
            };
            divide.Edges.Add(operand1);

            Edge operand2 = new Edge()
            {
                Relationship = RelationshipType.Denominator,
                TargetNode = functionWithCoefficent
            };
            divide.Edges.Add(operand2);

            graph.Nodes.Add(divide);

            return graph;
        }

        private GradientGraph Differentiate(GradientGraph graph)
        {

            return graph;
        }
    }
}