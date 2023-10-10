using GradientExplorer.ViewModels;
using System.Reflection;
using System.Windows.Controls;

namespace ToolWindow
{
    /// <summary>
    /// Interaction logic for GradientToolboxControl.xaml
    /// </summary>
    public partial class GradientToolboxControl : UserControl
    {
        public GradientToolboxControl(GradientToolboxViewModel viewModel)
        {
            AppDomain.CurrentDomain.AssemblyResolve += new ResolveEventHandler(MyResolver);
            InitializeComponent();
            DataContext = viewModel;
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
