using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace GradientExplorer.Parsers
{
    public interface IMethodParser
    {
        ExpressionSyntax ParseMethod(MethodDeclarationSyntax method);
    }
}
