using GradientExplorer.Model;
using Microsoft.CodeAnalysis;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public interface IExpressionDifferentiator
    {
        Node DifferentiateLiteral(SyntaxNode node, LiteralType type);
        GradientGraph DifferentiateExpExpression(SyntaxNode innerInvocation);
        GradientGraph DifferentiateSinExpression(SyntaxNode innerInvocation);
        GradientGraph DifferentiateSinhExpression(SyntaxNode innerInvocation);
        GradientGraph DifferentiateAsinExpression(SyntaxNode innerInvocation);
        GradientGraph DifferentiateCosExpression(SyntaxNode innerInvocation);
        GradientGraph DifferentiateCoshExpression(SyntaxNode innerInvocation);
        GradientGraph DifferentiateAcosExpression(SyntaxNode innerInvocation);
        GradientGraph DifferentiateTanExpression(SyntaxNode innerInvocation);
        GradientGraph DifferentiateTanhExpression(SyntaxNode innerInvocation);
        GradientGraph DifferentiateAtanExpression(SyntaxNode innerInvocation);
        GradientGraph DifferentiateLogExpression(SyntaxNode innerInvocation);
        GradientGraph DifferentiateLnExpression(SyntaxNode innerInvocation);
        GradientGraph DifferentiatePowExpression(List<SyntaxNode> syntaxNodes);
        GradientGraph DifferentiateSqrtExpression(SyntaxNode innerInvocation);
    }
}
