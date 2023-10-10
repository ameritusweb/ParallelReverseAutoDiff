using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using Autofac;
using Community.VisualStudio.Toolkit;
using GradientExplorer.Parsers;
using GradientExplorer.ViewModels;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Text;
using Microsoft.VisualStudio.Imaging;
using Microsoft.VisualStudio.Shell;

namespace ToolWindow
{
    public class GradientExplorer : BaseToolWindow<GradientExplorer>
    {
        private IContainer _container;

        public override string GetTitle(int toolWindowId) => "Gradient Explorer";
        public override Type PaneType => typeof(Pane);

        public override async Task<FrameworkElement> CreateAsync(int toolWindowId, CancellationToken cancellationToken)
        {
            Version vsVersion = await VS.Shell.GetVsVersionAsync();

            var builder = new ContainerBuilder();

            // Register your types here
            builder.RegisterType<MethodParser>().As<IMethodParser>();
            builder.RegisterType<GradientExplorerViewModel>().AsSelf();

            _container = builder.Build();

            var viewModel = _container.Resolve<GradientExplorerViewModel>();
            viewModel.VSVersion = vsVersion;

            return new GradientExplorerControl(viewModel);
        }

        [Guid("03030460-e1a2-49ab-a4c5-b7b9cfc2a4df")]
        public class Pane : ToolWindowPane
        {
            public Pane()
            {
                BitmapImageMoniker = KnownMonikers.ToolWindow;
            }
        }
    }
}