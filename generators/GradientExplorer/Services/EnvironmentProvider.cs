using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public class EnvironmentProvider : IEnvironmentProvider
    {
        public string GetNewLine()
        {
            return Environment.NewLine;
        }
    }
}
