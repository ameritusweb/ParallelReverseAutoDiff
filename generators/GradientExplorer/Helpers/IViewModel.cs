using GradientExplorer.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Helpers
{
    public interface IViewModel
    {
        public ILogger Logger { get; }
    }
}
