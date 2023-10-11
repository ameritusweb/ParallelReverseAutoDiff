using Microsoft.VisualStudio.Shell.Interop;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public interface IPaneCreator
    {
        IVsOutputWindowPane CreatePane(Guid paneGuid, string title, bool visible, bool clearWithSolution);
    }
}
