using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using Community.VisualStudio.Toolkit;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Text;
using Microsoft.VisualStudio.Imaging;
using Microsoft.VisualStudio.Shell;
using Microsoft.VisualStudio.Shell.Interop;

namespace ToolWindow
{
    public class GradientToolbox : BaseToolWindow<GradientToolbox>
    {
        public override string GetTitle(int toolWindowId) => "Gradient Toolbox";
        public override Type PaneType => typeof(Pane);

        public override async Task<FrameworkElement> CreateAsync(int toolWindowId, CancellationToken cancellationToken)
        {
            Version vsVersion = await VS.Shell.GetVsVersionAsync();
            return new GradientToolboxControl(vsVersion);
        }

        [Guid("03050460-e1a4-49ab-a3c5-b7b1cfc2b4da")]
        public class Pane : ToolWindowPane
        {
            public Pane()
            {
                BitmapImageMoniker = KnownMonikers.ToolWindow;
                ToolBarLocation = (int)VSTWT_LOCATION.VSTWT_LEFT;
            }
        }
    }
}