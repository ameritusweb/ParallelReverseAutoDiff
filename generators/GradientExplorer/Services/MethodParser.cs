using GradientExplorer.Model;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using GradientExplorer.Services;

namespace GradientExplorer.Parsers
{
    public class MethodParser : IMethodParser
    {
        public MethodParser()
        {
        }

        public ExpressionSyntax ParseMethod(MethodDeclarationSyntax method)
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

                                    return rightHandSide;
                                }
                            }
                        }
                    }
                }
            }
            return null;
        }
    }
}
