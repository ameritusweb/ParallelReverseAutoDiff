using GradientExplorer.LaTeX.Wpf;
using GradientExplorer.Model;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
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
                { NodeType.Sinh, DifferentiateSinhExpression },
                { NodeType.Asin, DifferentiateAsinExpression },
                { NodeType.Cos, DifferentiateCosExpression },
                { NodeType.Cosh, DifferentiateCoshExpression },
                { NodeType.Acos, DifferentiateAcosExpression },
                { NodeType.Tan, DifferentiateTanExpression },
                { NodeType.Tanh, DifferentiateTanhExpression },
                { NodeType.Atan, DifferentiateAtanExpression },
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
                var gradientGraph = await this.ParseMethod(forwardMethod);
                var canvas = laTeXCanvas;
                canvas.Children.Clear();
                var wpfCanvas = new WpfCanvas(canvas);
                WpfMathPainter painter = new WpfMathPainter();
                painter.LaTeX = gradientGraph.ToLaTeX();
                painter.Draw(wpfCanvas);
            }
        }

        private async Task<GradientGraph?> ParseMethod(MethodDeclarationSyntax method)
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
                                    GradientGraph gradientGraph = await DecomposeExpression(rightHandSide, new GradientGraph());
                                    return gradientGraph;
                                }
                            }
                        }
                    }
                }
            }
            return null;
        }

        private async Task<GradientGraph> DecomposeExpression(ExpressionSyntax expression, GradientGraph gradientGraph)
        {
            switch (expression)
            {
                case ParenthesizedExpressionSyntax parenthesizedExpression:
                    // Recursively decompose the inner expression
                    gradientGraph = await DecomposeExpression(parenthesizedExpression.Expression, gradientGraph);
                    break;
                case BinaryExpressionSyntax binaryExpression:
                    // Determine the rule based on the operator
                    if (binaryExpression.OperatorToken.Text == "+")
                    {
                        // Sum rule
                        SumRuleGradientExpression gradientExpression = await CreateSumRuleExpressionAsync(binaryExpression);
                        gradientGraph.Nodes.Add(gradientExpression.Differentiate());
                        gradientGraph.Expressions.Add(gradientExpression);
                    }
                    else if (binaryExpression.OperatorToken.Text == "-")
                    {
                        // Difference rule
                        DifferenceRuleGradientExpression gradientExpression = await CreateDifferenceRuleExpressionAsync(binaryExpression);
                        gradientGraph.Nodes.Add(gradientExpression.Differentiate());
                        gradientGraph.Expressions.Add(gradientExpression);
                    }
                    else if (binaryExpression.OperatorToken.Text == "*")
                    {
                        // Product rule
                        ProductRuleGradientExpression gradientExpression = await CreateProductRuleExpressionAsync(binaryExpression);
                        gradientGraph.Nodes.Add(gradientExpression.Differentiate());
                        gradientGraph.Expressions.Add(gradientExpression);
                    }
                    else if (binaryExpression.OperatorToken.Text == "/")
                    {
                        // Quotient rule
                        QuotientRuleGradientExpression gradientExpression = await CreateQuotientRuleExpressionAsync(binaryExpression);
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
                                CompositePowerRuleGradientExpression gradientExpression = await CreateCompositePowerRuleExpressionAsync(innerExpression, secondInnerExpression, innerInnerExpression, secondInnerInnerExpression, innerDifferentiationFunction, secondInnerDifferentiationFunction);
                                gradientGraph.Nodes.Add(gradientExpression.Differentiate());
                                gradientGraph.Expressions.Add(gradientExpression);
                            }
                        }
                        else if (secondInnerExpression is InvocationExpressionSyntax secondInnerInvocationExpression)
                        {
                            // Handle functions like Math.Pow(x, Math.Sin(x))
                            var secondInnerNodeType = GraphHelper.GetNodeType(secondInnerInvocationExpression);
                            var secondInnerDifferentiationFunction = gradientUnaryExpressionMap[secondInnerNodeType];
                            var secondInnerInnerExpression = secondInnerInvocationExpression.ArgumentList.Arguments[0].Expression;
                            CompositePowerRuleGradientExpression gradientExpression = await CreateCompositePowerRuleExpressionAsync(innerExpression, secondInnerExpression, secondInnerInnerExpression, secondInnerDifferentiationFunction);
                            gradientGraph.Nodes.Add(gradientExpression.Differentiate());
                            gradientGraph.Expressions.Add(gradientExpression);
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
                            if (gradientNonUnaryExpressionMap.ContainsKey(innerNodeType))
                            {
                                var innerDifferentiationFunction = gradientNonUnaryExpressionMap[innerNodeType];
                                var innerInnerExpressions = innerInvocationExpression.ArgumentList.Arguments.Select(x => x.Expression).ToList();
                                ChainRuleGradientExpression gradientExpression = await CreateNonUnaryChainRuleExpressionAsync(innerInvocationExpression, innerInnerExpressions, differentiationFunction, innerDifferentiationFunction);
                                gradientGraph.Nodes.Add(gradientExpression.Differentiate());
                                gradientGraph.Expressions.Add(gradientExpression);
                            }
                            else
                            {
                                var innerDifferentiationFunction = gradientUnaryExpressionMap[innerNodeType];
                                var innerInnerExpression = innerInvocationExpression.ArgumentList.Arguments[0].Expression;
                                ChainRuleGradientExpression gradientExpression = await CreateChainRuleExpressionAsync(innerInvocationExpression, innerInnerExpression, differentiationFunction, innerDifferentiationFunction);
                                gradientGraph.Nodes.Add(gradientExpression.Differentiate());
                                gradientGraph.Expressions.Add(gradientExpression);
                            }
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
                case PrefixUnaryExpressionSyntax prefixUnaryExpression:
                    gradientGraph.Nodes.Add(DifferentiateLiteral(prefixUnaryExpression, LiteralType.Variable));
                    break;
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

        private async Task<SumRuleGradientExpression> CreateSumRuleExpressionAsync(BinaryExpressionSyntax binaryExpression)
        {
            SumRuleGradientExpression gradientExpression = new SumRuleGradientExpression();

            // Recursively decompose left and right operands
            var left = Task.Run(() => DecomposeExpression(binaryExpression.Left, new GradientGraph()));
            var right = Task.Run(() => DecomposeExpression(binaryExpression.Right, new GradientGraph()));

            gradientExpression.Operands.Add(await left);
            gradientExpression.Operands.Add(await right);

            return gradientExpression;
        }

        private async Task<DifferenceRuleGradientExpression> CreateDifferenceRuleExpressionAsync(BinaryExpressionSyntax binaryExpression)
        {
            DifferenceRuleGradientExpression gradientExpression = new DifferenceRuleGradientExpression();

            // Recursively decompose left and right operands
            var left = Task.Run(() => DecomposeExpression(binaryExpression.Left, new GradientGraph()));
            var right = Task.Run(() => DecomposeExpression(binaryExpression.Right, new GradientGraph()));

            gradientExpression.Operands.Add(await left);
            gradientExpression.Operands.Add(await right);

            return gradientExpression;
        }

        private async Task<ChainRuleGradientExpression> CreateChainRuleExpressionAsync(ExpressionSyntax g, ExpressionSyntax innerG, Func<SyntaxNode, GradientGraph> diff, Func<SyntaxNode, GradientGraph> innerDiff)
        {
            ChainRuleGradientExpression gradientExpression = new ChainRuleGradientExpression();

            var taskFPrimeOfG = Task.Run(() => diff.Invoke(g));
            var taskGPrime = Task.Run(async () => IsDecomposable(innerG) ? await DecomposeExpression(g, new GradientGraph()) : await Task.FromResult(innerDiff.Invoke(innerG)));

            gradientExpression.FPrimeOfG = await taskFPrimeOfG;
            gradientExpression.GPrime = await taskGPrime;

            return gradientExpression;
        }

        private async Task<ChainRuleGradientExpression> CreateNonUnaryChainRuleExpressionAsync(ExpressionSyntax g, List<ExpressionSyntax> innerG, Func<SyntaxNode, GradientGraph> diff, Func<List<SyntaxNode>, GradientGraph> innerDiff)
        {
            ChainRuleGradientExpression gradientExpression = new ChainRuleGradientExpression();

            var taskFPrimeOfG = Task.Run(() => diff.Invoke(g));
            var taskGPrime = Task.Run(async () => innerG.Any(x => IsDecomposable(x)) ? await DecomposeExpression(g, new GradientGraph()) : await Task.FromResult(innerDiff.Invoke(innerG.Select(x => x as SyntaxNode).ToList())));

            gradientExpression.FPrimeOfG = await taskFPrimeOfG;
            gradientExpression.GPrime = await taskGPrime;

            return gradientExpression;
        }

        private async Task<CompositePowerRuleGradientExpression> CreateCompositePowerRuleExpressionAsync(ExpressionSyntax f, ExpressionSyntax g, ExpressionSyntax innerF, ExpressionSyntax innerG, Func<SyntaxNode, GradientGraph> diff1, Func<SyntaxNode, GradientGraph> diff2)
        {
            CompositePowerRuleGradientExpression gradientExpression = new CompositePowerRuleGradientExpression();

            var taskF = Task.Run(() => GraphHelper.ConvertToGraph(f));
            var taskG = Task.Run(() => GraphHelper.ConvertToGraph(g));
            var taskFPrime = Task.Run(async () => IsDecomposable(innerF) ? await DecomposeExpression(f, new GradientGraph()) : await Task.FromResult(diff1.Invoke(innerF)));
            var taskGPrime = Task.Run(async () => IsDecomposable(innerG) ? await DecomposeExpression(g, new GradientGraph()) : await Task.FromResult(diff2.Invoke(innerG)));

            // Await the tasks and continue
            gradientExpression.F = await taskF;
            gradientExpression.G = await taskG;
            gradientExpression.FPrime = await taskFPrime;
            gradientExpression.GPrime = await taskGPrime;

            return gradientExpression;
        }

        private async Task<CompositePowerRuleGradientExpression> CreateCompositePowerRuleExpressionAsync(ExpressionSyntax f, ExpressionSyntax g, ExpressionSyntax innerG, Func<SyntaxNode, GradientGraph> diff2)
        {
            CompositePowerRuleGradientExpression gradientExpression = new CompositePowerRuleGradientExpression();

            var taskF = Task.Run(() => GraphHelper.ConvertToGraph(f));
            var taskG = Task.Run(() => GraphHelper.ConvertToGraph(g));
            var taskFPrime = Task.Run(async () => await DecomposeExpression(f, new GradientGraph()));
            var taskGPrime = Task.Run(async () => IsDecomposable(innerG) ? await DecomposeExpression(g, new GradientGraph()) : await Task.FromResult(diff2.Invoke(innerG)));

            // Await the tasks and continue
            gradientExpression.F = await taskF;
            gradientExpression.G = await taskG;
            gradientExpression.FPrime = await taskFPrime;
            gradientExpression.GPrime = await taskGPrime;

            return gradientExpression;
        }

        private async Task<QuotientRuleGradientExpression> CreateQuotientRuleExpressionAsync(BinaryExpressionSyntax binaryExpression)
        {
            QuotientRuleGradientExpression gradientExpression = new QuotientRuleGradientExpression();

            // Define tasks for async execution
            var taskF = Task.Run(() => GraphHelper.ConvertToGraph(binaryExpression.Left));
            var taskG = Task.Run(() => GraphHelper.ConvertToGraph(binaryExpression.Right));
            var taskFPrime = Task.Run(() => DecomposeExpression(binaryExpression.Left, new GradientGraph()));
            var taskGPrime = Task.Run(() => DecomposeExpression(binaryExpression.Right, new GradientGraph()));

            // Await the tasks and continue
            gradientExpression.F = await taskF;
            gradientExpression.G = await taskG;
            gradientExpression.FPrime = await taskFPrime;
            gradientExpression.GPrime = await taskGPrime;

            return gradientExpression;
        }

        private async Task<ProductRuleGradientExpression> CreateProductRuleExpressionAsync(BinaryExpressionSyntax binaryExpression)
        {
            ProductRuleGradientExpression gradientExpression = new ProductRuleGradientExpression();

            // Define tasks for async execution
            var taskF = Task.Run(() => GraphHelper.ConvertToGraph(binaryExpression.Left));
            var taskG = Task.Run(() => GraphHelper.ConvertToGraph(binaryExpression.Right));
            var taskFPrime = Task.Run(() => DecomposeExpression(binaryExpression.Left, new GradientGraph()));
            var taskGPrime = Task.Run(() => DecomposeExpression(binaryExpression.Right, new GradientGraph()));

            // Await the tasks and continue
            gradientExpression.F = await taskF;
            gradientExpression.G = await taskG;
            gradientExpression.FPrime = await taskFPrime;
            gradientExpression.GPrime = await taskGPrime;

            return gradientExpression;
        }

        private bool IsDecomposable(SyntaxNode node)
        {
            return !(node is LiteralExpressionSyntax || node is IdentifierNameSyntax || node is ElementAccessExpressionSyntax);
        }

        private Node DifferentiateLiteral(SyntaxNode node, LiteralType type)
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

        private GradientGraph DifferentiateSinhExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var node = GraphHelper.Function(NodeType.Cosh, target);

            graph.Nodes.Add(node);
            return graph;
        }

        private GradientGraph DifferentiateAsinExpression(SyntaxNode innerInvocation)
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

        private GradientGraph DifferentiateCosExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node coefficient = GraphHelper.ToConstantNode(-1);

            var inner = GraphHelper.FunctionWithCoefficient(NodeType.Sin, coefficient, innerInvocation);

            graph.Nodes.Add(inner);

            return graph;
        }

        private GradientGraph DifferentiateCoshExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            var target = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var inner = GraphHelper.Function(NodeType.Sinh, target);

            graph.Nodes.Add(inner);

            return graph;
        }

        private GradientGraph DifferentiateAcosExpression(SyntaxNode innerInvocation)
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

        private GradientGraph DifferentiateTanExpression(SyntaxNode innerInvocation)
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

        private GradientGraph DifferentiateTanhExpression(SyntaxNode innerInvocation)
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

        private GradientGraph DifferentiateAtanExpression(SyntaxNode innerInvocation)
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

        private GradientGraph DifferentiateLogExpression(SyntaxNode innerInvocation)
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

        private GradientGraph DifferentiateLnExpression(SyntaxNode innerInvocation)
        {
            GradientGraph graph = new GradientGraph();

            Node numerator = GraphHelper.ToConstantNode(1);

            var denominator = GraphHelper.ConvertToGraph(innerInvocation).Nodes.FirstOrDefault();

            var divide = GraphHelper.Function(NodeType.Divide, numerator, denominator);

            graph.Nodes.Add(divide);

            return graph;
        }

        private GradientGraph DifferentiatePowExpression(List<SyntaxNode> syntaxNodes)
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

        private GradientGraph DifferentiateSqrtExpression(SyntaxNode innerInvocation)
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