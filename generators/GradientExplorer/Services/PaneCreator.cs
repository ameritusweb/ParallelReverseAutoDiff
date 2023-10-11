using Microsoft.VisualStudio.Shell.Interop;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;

namespace GradientExplorer.Services
{
    public class PaneCreator
    {
        public IVsOutputWindowPane CreatePane(Guid paneGuid, string title, bool visible, bool clearWithSolution)
        {
            ThreadHelper.ThrowIfNotOnUIThread();
            IVsOutputWindow outputWindow = (IVsOutputWindow)ServiceProvider.GlobalProvider.GetService(typeof(SVsOutputWindow));
            IVsOutputWindowPane pane;

            // Create a new pane.
            outputWindow.CreatePane(
                ref paneGuid,
                title,
                Convert.ToInt32(visible),
                Convert.ToInt32(clearWithSolution));

            // Retrieve the new pane.
            outputWindow.GetPane(ref paneGuid, out pane);
            return pane;
        }
    }
}
