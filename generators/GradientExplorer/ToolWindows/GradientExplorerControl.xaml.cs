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
        private Dictionary<FunctionType, Func<SyntaxNode, GradientGraph>> gradientUnaryExpressionMap;

        public GradientExplorerControl(Version vsVersion)
        {
            InitializeComponent();

            gradientUnaryExpressionMap = new Dictionary<FunctionType, Func<SyntaxNode, GradientGraph>>()
            {
                { FunctionType.Exp, DifferentiateExpExpression },
                { FunctionType.Sin, DifferentiateSinExpression },
                { FunctionType.Cos, DifferentiateCosExpression },
                { FunctionType.Tan, DifferentiateTanExpression },
                { FunctionType.Log, DifferentiateLogExpression },
                { FunctionType.Ln, DifferentiateLnExpression },
                { FunctionType.Sqrt, DifferentiateSqrtExpression },
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

                if (forwardMethod != null)
                {
                    var blockSyntax = forwardMethod.Body;
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
                    var functionType = this.GetFunctionType(invocationExpression);
                    if (functionType != FunctionType.Unknown)
                    {
                        var differentiationFunction = gradientUnaryExpressionMap[functionType];
                        var innerExpression = invocationExpression.ArgumentList.Arguments[0].Expression;
                        if (innerExpression is InvocationExpressionSyntax innerInvocationExpression)
                        {
                            // Handle nested functions like Math.Sin(Math.Cos(x))
                            var innerFunctionType = this.GetFunctionType(innerInvocationExpression);
                            var innerDifferentiationFunction = gradientUnaryExpressionMap[innerFunctionType];
                            var innerInnerExpression = innerInvocationExpression.ArgumentList.Arguments[0].Expression;
                            ChainRuleGradientExpression gradientExpression = new ChainRuleGradientExpression();
                            gradientExpression.FPrimeOfG = differentiationFunction.Invoke(innerInvocationExpression);
                            gradientExpression.GPrime = innerDifferentiationFunction.Invoke(innerInnerExpression);
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

        private FunctionType GetFunctionType(InvocationExpressionSyntax invocation)
        {
            if (invocation.Expression is MemberAccessExpressionSyntax memberAccessExpression)
            {
                if (memberAccessExpression.Expression is IdentifierNameSyntax identifierName)
                {
                    string name = identifierName.Identifier.Text + "." + memberAccessExpression.Name.Identifier.Text;
                    switch (name)
                    {
                        case "Math.Exp":
                            return FunctionType.Exp;
                        case "Math.Log":
                            return FunctionType.Ln;
                        case "Math.Log10":
                            return FunctionType.Log;
                        case "Math.Sin":
                            return FunctionType.Sin;
                        case "Math.Cos":
                            return FunctionType.Cos;
                        case "Math.Tan":
                            return FunctionType.Tan;
                        case "Math.Sqrt":
                            return FunctionType.Sqrt;
                        case "Math.Pow":
                            return FunctionType.Pow;
                        default:
                            throw new InvalidOperationException("Unknown function type");
                    }
                }
            }

            return FunctionType.Unknown;
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
                    FunctionType = FunctionType.Exp,
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
                    FunctionType = FunctionType.Cos,
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

            return graph;
        }

        private GradientGraph DifferentiateSqrtExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            return graph;
        }

        private GradientGraph Differentiate(GradientGraph graph)
        {

            return graph;
        }
    }
}