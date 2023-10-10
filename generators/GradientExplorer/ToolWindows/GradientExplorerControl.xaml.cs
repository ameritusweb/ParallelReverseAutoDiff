using FontAwesome.Sharp;
using GradientExplorer.Converters;
using GradientExplorer.Diagram;
using GradientExplorer.Extensions;
using GradientExplorer.LaTeX.Wpf;
using GradientExplorer.Model;
using GradientExplorer.ViewModels;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;
using Microsoft.VisualStudio.PlatformUI;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace ToolWindow
{
    public partial class GradientExplorerControl : UserControl
    {

        public GradientExplorerControl(GradientExplorerViewModel viewModel)
        {
            AppDomain.CurrentDomain.AssemblyResolve += new ResolveEventHandler(MyResolver);
            InitializeComponent();
            DataContext = viewModel;
            viewModel.UpdateBackgroundColor();
            viewModel.LoadData();
        }

        Assembly MyResolver(object sender, ResolveEventArgs args)
        {
            var e = Assembly.GetExecutingAssembly();
            if (args.Name == e.FullName)
            {
                return e;
            }
            return null;
        }
    }
}