using GradientExplorer.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public interface ILaTeXBuilder
    {
        StringBuilder GenerateLatexFromGraph(Node node, StringBuilder builder);
    }
}
