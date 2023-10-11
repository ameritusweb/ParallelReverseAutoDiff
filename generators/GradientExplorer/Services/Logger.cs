using GradientExplorer.Helpers;
using Microsoft.VisualStudio.Shell.Interop;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public class Logger : ILogger
    {
        private readonly IPaneCreator paneCreator;
        private IVsOutputWindowPane pane;

        public Logger(IPaneCreator paneCreator)
        {
            this.paneCreator = paneCreator;
            pane = paneCreator.CreatePane(Guid.NewGuid(), "Gradient Explorer", true, true);
        }

        public void Log(string message, SeverityType severity)
        {
            ThreadHelper.ThrowIfNotOnUIThread();

            string formattedMessage = $"[{DateTime.Now.ToString("HH:mm:ss")}] {severity.ToString().ToUpperInvariant()}: {message}";
            pane.OutputStringThreadSafe(formattedMessage + Environment.NewLine);
        }
    }
}
