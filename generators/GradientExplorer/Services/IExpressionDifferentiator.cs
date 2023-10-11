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
        Task<GradientGraph> DifferentiateExpExpressionAsync(SyntaxNode innerInvocation);
        Task<GradientGraph> DifferentiateSinExpressionAsync(SyntaxNode innerInvocation);
        Task<GradientGraph> DifferentiateSinhExpressionAsync(SyntaxNode innerInvocation);
        Task<GradientGraph> DifferentiateAsinExpressionAsync(SyntaxNode innerInvocation);
        Task<GradientGraph> DifferentiateCosExpressionAsync(SyntaxNode innerInvocation);
        Task<GradientGraph> DifferentiateCoshExpressionAsync(SyntaxNode innerInvocation);
        Task<GradientGraph> DifferentiateAcosExpressionAsync(SyntaxNode innerInvocation);
        Task<GradientGraph> DifferentiateTanExpressionAsync(SyntaxNode innerInvocation);
        Task<GradientGraph> DifferentiateTanhExpressionAsync(SyntaxNode innerInvocation);
        Task<GradientGraph> DifferentiateAtanExpressionAsync(SyntaxNode innerInvocation);
        Task<GradientGraph> DifferentiateLogExpressionAsync(SyntaxNode innerInvocation);
        Task<GradientGraph> DifferentiateLnExpressionAsync(SyntaxNode innerInvocation);
        Task<GradientGraph> DifferentiatePowExpressionAsync(List<SyntaxNode> syntaxNodes);
        Task<GradientGraph> DifferentiateSqrtExpressionAsync(SyntaxNode innerInvocation);
    }
}
