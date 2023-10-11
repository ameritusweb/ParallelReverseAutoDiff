using GradientExplorer.Model;
using GradientExplorer.Parsers;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public class GradientGraphFactory
    {
        private readonly IMethodParser methodParser;
        private readonly IExpressionDecomposer expressionDecomposer;

        public GradientGraphFactory(IMethodParser methodParser, IExpressionDecomposer expressionDecomposer)
        {
            this.methodParser = methodParser;
            this.expressionDecomposer = expressionDecomposer;
        }

        public async Task<GradientGraph> CreateGradientGraphAsync(MethodDeclarationSyntax methodSyntax)
        {
            var rightHandSide = methodParser.ParseMethod(methodSyntax);
            // Decompose the right-hand side into a gradient graph
            GradientGraph gradientGraph = await expressionDecomposer.DecomposeExpressionAsync(rightHandSide, new GradientGraph());
            return gradientGraph;
        }
    }
}
