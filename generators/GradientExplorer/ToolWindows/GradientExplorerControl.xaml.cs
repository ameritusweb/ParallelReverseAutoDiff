using GradientExplorer.Diagram;
using GradientExplorer.LaTeX.Wpf;
using GradientExplorer.Model;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;
using Microsoft.VisualStudio.PlatformUI;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace ToolWindow
{
    public partial class GradientExplorerControl : UserControl
    {
        private Dictionary<NodeType, Func<SyntaxNode, GradientGraph>> gradientUnaryExpressionMap;
        private Dictionary<NodeType, Func<List<SyntaxNode>, GradientGraph>> gradientNonUnaryExpressionMap;
        private ConcurrentDictionary<string, GradientGraph> gradientCache;
        private DiagramCanvas currentDiagram;
        private WpfMathPainter painter;

        public GradientExplorerControl(Version vsVersion)
        {
            InitializeComponent();

            gradientUnaryExpressionMap = new Dictionary<NodeType, Func<SyntaxNode, GradientGraph>>()
            {
                { NodeType.Exp, DifferentiationHelper.DifferentiateExpExpression },
                { NodeType.Sin, DifferentiationHelper.DifferentiateSinExpression },
                { NodeType.Sinh, DifferentiationHelper.DifferentiateSinhExpression },
                { NodeType.Asin, DifferentiationHelper.DifferentiateAsinExpression },
                { NodeType.Cos, DifferentiationHelper.DifferentiateCosExpression },
                { NodeType.Cosh, DifferentiationHelper.DifferentiateCoshExpression },
                { NodeType.Acos, DifferentiationHelper.DifferentiateAcosExpression },
                { NodeType.Tan, DifferentiationHelper.DifferentiateTanExpression },
                { NodeType.Tanh, DifferentiationHelper.DifferentiateTanhExpression },
                { NodeType.Atan, DifferentiationHelper.DifferentiateAtanExpression },
                { NodeType.Log, DifferentiationHelper.DifferentiateLogExpression },
                { NodeType.Ln, DifferentiationHelper.DifferentiateLnExpression },
                { NodeType.Sqrt, DifferentiationHelper.DifferentiateSqrtExpression },
            };

            gradientNonUnaryExpressionMap = new Dictionary<NodeType, Func<List<SyntaxNode>, GradientGraph>>()
            {
                { NodeType.Pow, DifferentiationHelper.DifferentiatePowExpression },
                // Add other non-unary functions here
            };

            gradientCache = new ConcurrentDictionary<string, GradientGraph>();

            VSColorTheme.ThemeChanged += VSColorTheme_ThemeChanged;

            lblHeadline.Content = $"Visual Studio v{vsVersion}";

            var backgroundColor = VSColorTheme.GetThemedColor(EnvironmentColors.ToolWindowBackgroundColorKey);
            rootPanel.Background = new SolidColorBrush(System.Windows.Media.Color.FromArgb(backgroundColor.A, backgroundColor.R, backgroundColor.G, backgroundColor.B));
        }

        private async void VSColorTheme_ThemeChanged(ThemeChangedEventArgs e)
        {
            var backgroundColor = VSColorTheme.GetThemedColor(EnvironmentColors.ToolWindowBackgroundColorKey);
            rootPanel.Background = new SolidColorBrush(System.Windows.Media.Color.FromArgb(backgroundColor.A, backgroundColor.R, backgroundColor.G, backgroundColor.B));

            var currentTheme = await ThemeManager.Instance.GetCurrentThemeAsync();
            if (this.currentDiagram != null)
            {
                bool changed = this.currentDiagram.UpdateTheme(currentTheme);
                if (changed && painter != null)
                {
                    laTeXCanvas.Children.Clear();
                    var wpfCanvas = new WpfCanvas(laTeXCanvas);
                    painter.Draw(wpfCanvas);
                }
            }
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
                painter = new WpfMathPainter();
                painter.LaTeX = gradientGraph.ToLaTeX();
                lblHeadline.Content = painter.LaTeX;
                painter.Draw(wpfCanvas);
                var currentTheme = await ThemeManager.Instance.GetCurrentThemeAsync();
                if (currentDiagram == null)
                {
                    currentDiagram = new DiagramCanvas(gradientGraph.DeepCopy(), currentTheme);
                    currentDiagram.BuildGraph();
                    var panel = currentDiagram.ToPanel();
                    ScaleTransform flipTransform = new ScaleTransform(1, -1);
                    panel.LayoutTransform = flipTransform;
                    mainPanel.Children.Add(panel);
                }
                else
                {
                    currentDiagram.Reinitialize(gradientGraph.DeepCopy(), currentTheme);
                    currentDiagram.BuildGraph();
                }
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
            string expressionFingerprint = expression.ToFullString();

            // Check if the gradient for this expression is cached
            if (gradientCache.ContainsKey(expressionFingerprint))
            {
                return gradientCache[expressionFingerprint].DeepCopy();
            }

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
                                if (gradientNonUnaryExpressionMap.ContainsKey(innerNodeType))
                                {
                                    var innerDifferentiationFunction = gradientNonUnaryExpressionMap[innerNodeType];
                                    var innerInnerExpressions = innerInvocationExpression.ArgumentList.Arguments.Select(x => x.Expression).ToList();
                                    var secondInnerNodeType = GraphHelper.GetNodeType(secondInnerInvocationExpression);
                                    var secondInnerDifferentiationFunction = gradientUnaryExpressionMap[secondInnerNodeType];
                                    var secondInnerInnerExpression = secondInnerInvocationExpression.ArgumentList.Arguments[0].Expression;
                                    CompositePowerRuleGradientExpression gradientExpression = await CreateNonUnaryCompositePowerRuleExpressionAsync(innerExpression, secondInnerExpression, innerInnerExpressions, secondInnerInnerExpression, innerDifferentiationFunction, secondInnerDifferentiationFunction);
                                    gradientGraph.Nodes.Add(gradientExpression.Differentiate());
                                    gradientGraph.Expressions.Add(gradientExpression);
                                }
                                else
                                {
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
                    gradientGraph.Nodes.Add(DifferentiationHelper.DifferentiateLiteral(prefixUnaryExpression, LiteralType.Variable));
                    break;
                case ElementAccessExpressionSyntax elementAccess:
                    gradientGraph.Nodes.Add(DifferentiationHelper.DifferentiateLiteral(elementAccess, LiteralType.Variable));
                    break;
                case LiteralExpressionSyntax literalExpression:
                    gradientGraph.Nodes.Add(DifferentiationHelper.DifferentiateLiteral(literalExpression, LiteralType.Constant));
                    break;

                default:
                    break;
            }

            // Cache the computed gradient before returning it
            gradientCache.AddOrUpdate(expressionFingerprint, gradientGraph, (key, oldValue) => gradientGraph);

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

        private async Task<CompositePowerRuleGradientExpression> CreateNonUnaryCompositePowerRuleExpressionAsync(ExpressionSyntax f, ExpressionSyntax g, List<ExpressionSyntax> innerF, ExpressionSyntax innerG, Func<List<SyntaxNode>, GradientGraph> diff1, Func<SyntaxNode, GradientGraph> diff2)
        {
            CompositePowerRuleGradientExpression gradientExpression = new CompositePowerRuleGradientExpression();

            var taskF = Task.Run(() => GraphHelper.ConvertToGraph(f));
            var taskG = Task.Run(() => GraphHelper.ConvertToGraph(g));
            var taskFPrime = Task.Run(async () => innerF.Any(x => IsDecomposable(x)) ? await DecomposeExpression(f, new GradientGraph()) : await Task.FromResult(diff1.Invoke(innerF.Select(x => x as SyntaxNode).ToList())));
            var taskGPrime = Task.Run(async () => IsDecomposable(innerG) ? await DecomposeExpression(g, new GradientGraph()) : await Task.FromResult(diff2.Invoke(innerG)));

            // Await the tasks and continue
            gradientExpression.F = await taskF;
            gradientExpression.G = await taskG;
            gradientExpression.FPrime = await taskFPrime;
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
    }
}