using GradientExplorer.Model;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public interface IGradientGraphFactory
    {
        Task<GradientGraph> CreateGradientGraphAsync(MethodDeclarationSyntax methodSyntax);
    }
}
