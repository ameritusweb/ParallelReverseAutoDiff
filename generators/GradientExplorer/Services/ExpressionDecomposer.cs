using GradientExplorer.Model;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Concurrent;

namespace GradientExplorer.Services
{
    public class ExpressionDecomposer : IExpressionDecomposer
    {
        private readonly Dictionary<NodeType, Func<SyntaxNode, GradientGraph>> gradientUnaryExpressionMap;
        private readonly Dictionary<NodeType, Func<List<SyntaxNode>, GradientGraph>> gradientNonUnaryExpressionMap;
        private readonly ConcurrentDictionary<string, GradientGraph> gradientCache;
        private INodeFactory nodeFactory;
        private INodeTypeFactory nodeTypeFactory;
        private IExpressionDifferentiator expressionDifferentiator;

        public ExpressionDecomposer(INodeFactory nodeFactory, INodeTypeFactory nodeTypeFactory, IExpressionDifferentiator expressionDifferentiator)
        {
            this.nodeFactory = nodeFactory;
            this.nodeTypeFactory = nodeTypeFactory;
            this.expressionDifferentiator = expressionDifferentiator;
            gradientUnaryExpressionMap = new Dictionary<NodeType, Func<SyntaxNode, GradientGraph>>()
            {
                { NodeType.Exp, expressionDifferentiator.DifferentiateExpExpression },
                { NodeType.Sin, expressionDifferentiator.DifferentiateSinExpression },
                { NodeType.Sinh, expressionDifferentiator.DifferentiateSinhExpression },
                { NodeType.Asin, expressionDifferentiator.DifferentiateAsinExpression },
                { NodeType.Cos, expressionDifferentiator.DifferentiateCosExpression },
                { NodeType.Cosh, expressionDifferentiator.DifferentiateCoshExpression },
                { NodeType.Acos, expressionDifferentiator.DifferentiateAcosExpression },
                { NodeType.Tan, expressionDifferentiator.DifferentiateTanExpression },
                { NodeType.Tanh, expressionDifferentiator.DifferentiateTanhExpression },
                { NodeType.Atan, expressionDifferentiator.DifferentiateAtanExpression },
                { NodeType.Log, expressionDifferentiator.DifferentiateLogExpression },
                { NodeType.Ln, expressionDifferentiator.DifferentiateLnExpression },
                { NodeType.Sqrt, expressionDifferentiator.DifferentiateSqrtExpression },
            };

            gradientNonUnaryExpressionMap = new Dictionary<NodeType, Func<List<SyntaxNode>, GradientGraph>>()
            {
                { NodeType.Pow, expressionDifferentiator.DifferentiatePowExpression },
                // Add other non-unary functions here
            };

            gradientCache = new ConcurrentDictionary<string, GradientGraph>();
            this.expressionDifferentiator = expressionDifferentiator;
        }

        public async Task<GradientGraph> DecomposeExpressionAsync(ExpressionSyntax expression, GradientGraph gradientGraph)
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
                    gradientGraph = await DecomposeExpressionAsync(parenthesizedExpression.Expression, gradientGraph);
                    break;
                case BinaryExpressionSyntax binaryExpression:
                    // Determine the rule based on the operator
                    if (binaryExpression.OperatorToken.Text == "+")
                    {
                        // Sum rule
                        SumRuleGradientExpression gradientExpression = await CreateSumRuleExpressionAsync(binaryExpression);
                        gradientGraph.Nodes.Add(gradientExpression.Differentiate(nodeFactory));
                        gradientGraph.Expressions.Add(gradientExpression);
                    }
                    else if (binaryExpression.OperatorToken.Text == "-")
                    {
                        // Difference rule
                        DifferenceRuleGradientExpression gradientExpression = await CreateDifferenceRuleExpressionAsync(binaryExpression);
                        gradientGraph.Nodes.Add(gradientExpression.Differentiate(nodeFactory));
                        gradientGraph.Expressions.Add(gradientExpression);
                    }
                    else if (binaryExpression.OperatorToken.Text == "*")
                    {
                        // Product rule
                        ProductRuleGradientExpression gradientExpression = await CreateProductRuleExpressionAsync(binaryExpression);
                        gradientGraph.Nodes.Add(gradientExpression.Differentiate(nodeFactory));
                        gradientGraph.Expressions.Add(gradientExpression);
                    }
                    else if (binaryExpression.OperatorToken.Text == "/")
                    {
                        // Quotient rule
                        QuotientRuleGradientExpression gradientExpression = await CreateQuotientRuleExpressionAsync(binaryExpression);
                        gradientGraph.Nodes.Add(gradientExpression.Differentiate(nodeFactory));
                        gradientGraph.Expressions.Add(gradientExpression);
                    }

                    break;

                case InvocationExpressionSyntax invocationExpression:
                    // Handle functions like Math.Exp, etc.
                    var functionType = nodeTypeFactory.GetNodeType(invocationExpression);
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
                                var innerNodeType = nodeTypeFactory.GetNodeType(innerInvocationExpression);
                                if (gradientNonUnaryExpressionMap.ContainsKey(innerNodeType))
                                {
                                    var innerDifferentiationFunction = gradientNonUnaryExpressionMap[innerNodeType];
                                    var innerInnerExpressions = innerInvocationExpression.ArgumentList.Arguments.Select(x => x.Expression).ToList();
                                    var secondInnerNodeType = nodeTypeFactory.GetNodeType(secondInnerInvocationExpression);
                                    var secondInnerDifferentiationFunction = gradientUnaryExpressionMap[secondInnerNodeType];
                                    var secondInnerInnerExpression = secondInnerInvocationExpression.ArgumentList.Arguments[0].Expression;
                                    CompositePowerRuleGradientExpression gradientExpression = await CreateNonUnaryCompositePowerRuleExpressionAsync(innerExpression, secondInnerExpression, innerInnerExpressions, secondInnerInnerExpression, innerDifferentiationFunction, secondInnerDifferentiationFunction);
                                    gradientGraph.Nodes.Add(gradientExpression.Differentiate(nodeFactory));
                                    gradientGraph.Expressions.Add(gradientExpression);
                                }
                                else
                                {
                                    var innerDifferentiationFunction = gradientUnaryExpressionMap[innerNodeType];
                                    var innerInnerExpression = innerInvocationExpression.ArgumentList.Arguments[0].Expression;
                                    var secondInnerNodeType = nodeTypeFactory.GetNodeType(secondInnerInvocationExpression);
                                    var secondInnerDifferentiationFunction = gradientUnaryExpressionMap[secondInnerNodeType];
                                    var secondInnerInnerExpression = secondInnerInvocationExpression.ArgumentList.Arguments[0].Expression;
                                    CompositePowerRuleGradientExpression gradientExpression = await CreateCompositePowerRuleExpressionAsync(innerExpression, secondInnerExpression, innerInnerExpression, secondInnerInnerExpression, innerDifferentiationFunction, secondInnerDifferentiationFunction);
                                    gradientGraph.Nodes.Add(gradientExpression.Differentiate(nodeFactory));
                                    gradientGraph.Expressions.Add(gradientExpression);
                                }
                            }
                        }
                        else if (secondInnerExpression is InvocationExpressionSyntax secondInnerInvocationExpression)
                        {
                            // Handle functions like Math.Pow(x, Math.Sin(x))
                            var secondInnerNodeType = nodeTypeFactory.GetNodeType(secondInnerInvocationExpression);
                            var secondInnerDifferentiationFunction = gradientUnaryExpressionMap[secondInnerNodeType];
                            var secondInnerInnerExpression = secondInnerInvocationExpression.ArgumentList.Arguments[0].Expression;
                            CompositePowerRuleGradientExpression gradientExpression = await CreateCompositePowerRuleExpressionAsync(innerExpression, secondInnerExpression, secondInnerInnerExpression, secondInnerDifferentiationFunction);
                            gradientGraph.Nodes.Add(gradientExpression.Differentiate(nodeFactory));
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
                            var innerNodeType = nodeTypeFactory.GetNodeType(innerInvocationExpression);
                            if (gradientNonUnaryExpressionMap.ContainsKey(innerNodeType))
                            {
                                var innerDifferentiationFunction = gradientNonUnaryExpressionMap[innerNodeType];
                                var innerInnerExpressions = innerInvocationExpression.ArgumentList.Arguments.Select(x => x.Expression).ToList();
                                ChainRuleGradientExpression gradientExpression = await CreateNonUnaryChainRuleExpressionAsync(innerInvocationExpression, innerInnerExpressions, differentiationFunction, innerDifferentiationFunction);
                                gradientGraph.Nodes.Add(gradientExpression.Differentiate(nodeFactory));
                                gradientGraph.Expressions.Add(gradientExpression);
                            }
                            else
                            {
                                var innerDifferentiationFunction = gradientUnaryExpressionMap[innerNodeType];
                                var innerInnerExpression = innerInvocationExpression.ArgumentList.Arguments[0].Expression;
                                ChainRuleGradientExpression gradientExpression = await CreateChainRuleExpressionAsync(innerInvocationExpression, innerInnerExpression, differentiationFunction, innerDifferentiationFunction);
                                gradientGraph.Nodes.Add(gradientExpression.Differentiate(nodeFactory));
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
                    gradientGraph.Nodes.Add(expressionDifferentiator.DifferentiateLiteral(prefixUnaryExpression, LiteralType.Variable));
                    break;
                case ElementAccessExpressionSyntax elementAccess:
                    gradientGraph.Nodes.Add(expressionDifferentiator.DifferentiateLiteral(elementAccess, LiteralType.Variable));
                    break;
                case LiteralExpressionSyntax literalExpression:
                    gradientGraph.Nodes.Add(expressionDifferentiator.DifferentiateLiteral(literalExpression, LiteralType.Constant));
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
            var left = Task.Run(() => DecomposeExpressionAsync(binaryExpression.Left, new GradientGraph()));
            var right = Task.Run(() => DecomposeExpressionAsync(binaryExpression.Right, new GradientGraph()));

            gradientExpression.Operands.Add(await left);
            gradientExpression.Operands.Add(await right);

            return gradientExpression;
        }

        private async Task<DifferenceRuleGradientExpression> CreateDifferenceRuleExpressionAsync(BinaryExpressionSyntax binaryExpression)
        {
            DifferenceRuleGradientExpression gradientExpression = new DifferenceRuleGradientExpression();

            // Recursively decompose left and right operands
            var left = Task.Run(() => DecomposeExpressionAsync(binaryExpression.Left, new GradientGraph()));
            var right = Task.Run(() => DecomposeExpressionAsync(binaryExpression.Right, new GradientGraph()));

            gradientExpression.Operands.Add(await left);
            gradientExpression.Operands.Add(await right);

            return gradientExpression;
        }

        private async Task<ChainRuleGradientExpression> CreateChainRuleExpressionAsync(ExpressionSyntax g, ExpressionSyntax innerG, Func<SyntaxNode, GradientGraph> diff, Func<SyntaxNode, GradientGraph> innerDiff)
        {
            ChainRuleGradientExpression gradientExpression = new ChainRuleGradientExpression();

            var taskFPrimeOfG = Task.Run(() => diff.Invoke(g));
            var taskGPrime = Task.Run(async () => IsDecomposable(innerG) ? await DecomposeExpressionAsync(g, new GradientGraph()) : await Task.FromResult(innerDiff.Invoke(innerG)));

            gradientExpression.FPrimeOfG = await taskFPrimeOfG;
            gradientExpression.GPrime = await taskGPrime;

            return gradientExpression;
        }

        private async Task<ChainRuleGradientExpression> CreateNonUnaryChainRuleExpressionAsync(ExpressionSyntax g, List<ExpressionSyntax> innerG, Func<SyntaxNode, GradientGraph> diff, Func<List<SyntaxNode>, GradientGraph> innerDiff)
        {
            ChainRuleGradientExpression gradientExpression = new ChainRuleGradientExpression();

            var taskFPrimeOfG = Task.Run(() => diff.Invoke(g));
            var taskGPrime = Task.Run(async () => innerG.Any(x => IsDecomposable(x)) ? await DecomposeExpressionAsync(g, new GradientGraph()) : await Task.FromResult(innerDiff.Invoke(innerG.Select(x => x as SyntaxNode).ToList())));

            gradientExpression.FPrimeOfG = await taskFPrimeOfG;
            gradientExpression.GPrime = await taskGPrime;

            return gradientExpression;
        }

        private async Task<CompositePowerRuleGradientExpression> CreateNonUnaryCompositePowerRuleExpressionAsync(ExpressionSyntax f, ExpressionSyntax g, List<ExpressionSyntax> innerF, ExpressionSyntax innerG, Func<List<SyntaxNode>, GradientGraph> diff1, Func<SyntaxNode, GradientGraph> diff2)
        {
            CompositePowerRuleGradientExpression gradientExpression = new CompositePowerRuleGradientExpression();

            var taskF = Task.Run(() => nodeFactory.ConvertToGraph(f));
            var taskG = Task.Run(() => nodeFactory.ConvertToGraph(g));
            var taskFPrime = Task.Run(async () => innerF.Any(x => IsDecomposable(x)) ? await DecomposeExpressionAsync(f, new GradientGraph()) : await Task.FromResult(diff1.Invoke(innerF.Select(x => x as SyntaxNode).ToList())));
            var taskGPrime = Task.Run(async () => IsDecomposable(innerG) ? await DecomposeExpressionAsync(g, new GradientGraph()) : await Task.FromResult(diff2.Invoke(innerG)));

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

            var taskF = Task.Run(() => nodeFactory.ConvertToGraph(f));
            var taskG = Task.Run(() => nodeFactory.ConvertToGraph(g));
            var taskFPrime = Task.Run(async () => IsDecomposable(innerF) ? await DecomposeExpressionAsync(f, new GradientGraph()) : await Task.FromResult(diff1.Invoke(innerF)));
            var taskGPrime = Task.Run(async () => IsDecomposable(innerG) ? await DecomposeExpressionAsync(g, new GradientGraph()) : await Task.FromResult(diff2.Invoke(innerG)));

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

            var taskF = Task.Run(() => nodeFactory.ConvertToGraph(f));
            var taskG = Task.Run(() => nodeFactory.ConvertToGraph(g));
            var taskFPrime = Task.Run(async () => await DecomposeExpressionAsync(f, new GradientGraph()));
            var taskGPrime = Task.Run(async () => IsDecomposable(innerG) ? await DecomposeExpressionAsync(g, new GradientGraph()) : await Task.FromResult(diff2.Invoke(innerG)));

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
            var taskF = Task.Run(() => nodeFactory.ConvertToGraph(binaryExpression.Left));
            var taskG = Task.Run(() => nodeFactory.ConvertToGraph(binaryExpression.Right));
            var taskFPrime = Task.Run(() => DecomposeExpressionAsync(binaryExpression.Left, new GradientGraph()));
            var taskGPrime = Task.Run(() => DecomposeExpressionAsync(binaryExpression.Right, new GradientGraph()));

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
            var taskF = Task.Run(() => nodeFactory.ConvertToGraph(binaryExpression.Left));
            var taskG = Task.Run(() => nodeFactory.ConvertToGraph(binaryExpression.Right));
            var taskFPrime = Task.Run(() => DecomposeExpressionAsync(binaryExpression.Left, new GradientGraph()));
            var taskGPrime = Task.Run(() => DecomposeExpressionAsync(binaryExpression.Right, new GradientGraph()));

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
