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
                var gradientGraph = this.ParseMethod(forwardMethod);
                var canvas = laTeXCanvas;
                var wpfCanvas = new WpfCanvas(canvas);
                WpfMathPainter painter = new WpfMathPainter();
                painter.LaTeX = gradientGraph.ToLaTeX();
                painter.Draw(wpfCanvas);
            }
        }

        private GradientGraph? ParseMethod(MethodDeclarationSyntax method)
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
                                    return gradientGraph;
                                }
                            }
                        }
                    }
                }
            }
            return null;
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
                        gradientExpression.Operands.Add(DecomposeExpression(binaryExpression.Left, new GradientGraph()));
                        gradientExpression.Operands.Add(DecomposeExpression(binaryExpression.Right, new GradientGraph()));
                        gradientGraph.Nodes.Add(gradientExpression.Differentiate());
                        gradientGraph.Expressions.Add(gradientExpression);
                    }
                    else if (binaryExpression.OperatorToken.Text == "*")
                    {
                        // Product rule
                        ProductRuleGradientExpression gradientExpression = new ProductRuleGradientExpression();
                        // Recursively decompose left and right operands
                        gradientExpression.F = GraphHelper.ConvertToGraph(binaryExpression.Left);
                        gradientExpression.G = GraphHelper.ConvertToGraph(binaryExpression.Right);
                        gradientExpression.FPrime = DecomposeExpression(binaryExpression.Left, new GradientGraph());
                        gradientExpression.GPrime = DecomposeExpression(binaryExpression.Right, new GradientGraph());
                        gradientGraph.Nodes.Add(gradientExpression.Differentiate());
                        gradientGraph.Expressions.Add(gradientExpression);
                    }
                    else if (binaryExpression.OperatorToken.Text == "/")
                    {
                        // Quotient rule
                        QuotientRuleGradientExpression gradientExpression = new QuotientRuleGradientExpression();
                        // Recursively decompose left and right operands
                        gradientExpression.F = GraphHelper.ConvertToGraph(binaryExpression.Left);
                        gradientExpression.G = GraphHelper.ConvertToGraph(binaryExpression.Right);
                        gradientExpression.FPrime = DecomposeExpression(binaryExpression.Left, new GradientGraph());
                        gradientExpression.GPrime = DecomposeExpression(binaryExpression.Right, new GradientGraph());
                        gradientGraph.Nodes.Add(gradientExpression.Differentiate());
                        gradientGraph.Expressions.Add(gradientExpression);
                    }

                    break;

                case InvocationExpressionSyntax invocationExpression:
                    // Handle functions like Math.Exp, etc.
                    var functionType = GraphHelper.GetNodeType(invocationExpression);
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
                                var innerNodeType = GraphHelper.GetNodeType(innerInvocationExpression);
                                var innerDifferentiationFunction = gradientUnaryExpressionMap[innerNodeType];
                                var innerInnerExpression = innerInvocationExpression.ArgumentList.Arguments[0].Expression;
                                var secondInnerNodeType = GraphHelper.GetNodeType(secondInnerInvocationExpression);
                                var secondInnerDifferentiationFunction = gradientUnaryExpressionMap[secondInnerNodeType];
                                var secondInnerInnerExpression = secondInnerInvocationExpression.ArgumentList.Arguments[0].Expression;
                                CompositePowerRuleGradientExpression gradientExpression = new CompositePowerRuleGradientExpression();
                                gradientExpression.F = GraphHelper.ConvertToGraph(innerInnerExpression);
                                gradientExpression.G = GraphHelper.ConvertToGraph(secondInnerInvocationExpression);
                                gradientExpression.FPrime = IsDecomposable(innerInnerExpression) ? DecomposeExpression(innerInnerExpression, new GradientGraph()) : innerDifferentiationFunction.Invoke(innerInnerExpression);
                                gradientExpression.GPrime = IsDecomposable(secondInnerInnerExpression) ? DecomposeExpression(secondInnerInnerExpression, new GradientGraph()) : secondInnerDifferentiationFunction.Invoke(secondInnerInnerExpression);
                                gradientGraph.Nodes.Add(gradientExpression.Differentiate());
                                gradientGraph.Expressions.Add(gradientExpression);
                            }
                        }
                        else
                        {
                            // Handle functions like Math.Pow(x, 2)
                            UnaryGradientExpression gradientExpression = new UnaryGradientExpression();
                            gradientExpression.FPrime = differentiationFunction.Invoke(new List<SyntaxNode>() { innerExpression, secondInnerExpression });
                            gradientGraph.Nodes.Add(gradientExpression.Differentiate());
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
                            var innerNodeType = GraphHelper.GetNodeType(innerInvocationExpression);
                            var innerDifferentiationFunction = gradientUnaryExpressionMap[innerNodeType];
                            var innerInnerExpression = innerInvocationExpression.ArgumentList.Arguments[0].Expression;
                            ChainRuleGradientExpression gradientExpression = new ChainRuleGradientExpression();
                            gradientExpression.FPrimeOfG = differentiationFunction.Invoke(innerInvocationExpression);
                            gradientExpression.GPrime = IsDecomposable(innerInnerExpression) ? DecomposeExpression(innerInvocationExpression, new GradientGraph()) : innerDifferentiationFunction.Invoke(innerInnerExpression);
                            gradientGraph.Nodes.Add(gradientExpression.Differentiate());
                            gradientGraph.Expressions.Add(gradientExpression);
                        }
                        else
                        {
                            // Handle functions like Math.Exp(x)
                            UnaryGradientExpression gradientExpression = new UnaryGradientExpression();
                            gradientExpression.FPrime = differentiationFunction.Invoke(innerExpression);
                            gradientGraph.Nodes.Add(gradientExpression.Differentiate());
                            gradientGraph.Expressions.Add(gradientExpression);
                        }
                    }
                    break;

                // ... handle other types of expressions

                // Base cases: expression cannot be decomposed further (e.g., literals, identifiers)
                case ElementAccessExpressionSyntax elementAccess:
                    gradientGraph.Nodes.Add(DifferentiateLiteral(elementAccess, LiteralType.Variable));
                    break;
                case LiteralExpressionSyntax literalExpression:
                    gradientGraph.Nodes.Add(DifferentiateLiteral(literalExpression, LiteralType.Constant));
                    break;

                default:
                    break;
            }

            return gradientGraph;
        }

        private bool IsDecomposable(SyntaxNode node)
        {
            return !(node is LiteralExpressionSyntax || node is IdentifierNameSyntax || node is ElementAccessExpressionSyntax);
        }

        private Node DifferentiateLiteral(SyntaxNode node, LiteralType type)
        {
            Node literalNode = new Node(node, node.GetType());
            literalNode.Value = type == LiteralType.Constant ? 0 : 1;
            literalNode.Type = type.ToString();
            return literalNode;
        }

        private GradientGraph DifferentiateExpExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();
            var target = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var node = GraphHelper.Function(NodeType.Exp, target);

            graph.Nodes.Add(node);
            return graph;
        }

        private GradientGraph DifferentiateSinExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var node = GraphHelper.Function(NodeType.Cos, target);

            graph.Nodes.Add(node);
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
                Type = LiteralType.Constant.ToString(),
            };

            var denominator = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

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

        private GradientGraph DifferentiatePowExpression(List<SyntaxNode> syntaxNodes)
        {
            GradientGraph graph = new GradientGraph();

            var baseNode = GraphHelper.ConvertToGraph(syntaxNodes[0]).Nodes.FirstOrDefault();

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
                Type = LiteralType.Constant.ToString(),
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
                Type = LiteralType.Constant.ToString(),
            };

            var functionWithCoefficent = GraphHelper.FunctionWithCoefficient(NodeType.Sqrt, two, innerInvocation);

            Node divide = new Node()
            {
                NodeType = NodeType.Divide
            };

            Node numerator = new Node()
            {
                Value = 1,
                Type = LiteralType.Constant.ToString(),
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
    }
}