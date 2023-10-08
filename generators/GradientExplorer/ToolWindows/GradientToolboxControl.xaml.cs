using FontAwesome.Sharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace ToolWindow
{
    /// <summary>
    /// Interaction logic for GradientToolboxControl.xaml
    /// </summary>
    public partial class GradientToolboxControl : UserControl
    {
        public GradientToolboxControl(Version vsVersion)
        {
            InitializeComponent();

            IconImage wandimage = new IconImage()
            {
                Icon = IconChar.WandMagicSparkles,
                Foreground = Brushes.CornflowerBlue
            };
            simplification.Children.Insert(0, wandimage);

            IconImage computationimage = new IconImage()
            {
                Icon = IconChar.DiagramProject,
                Foreground = Brushes.CornflowerBlue
            };
            computation.Children.Insert(0, computationimage);

            IconImage playimage = new IconImage()
            {
                Icon = IconChar.Play,
                Foreground = Brushes.CornflowerBlue
            };
            debugging.Children.Insert(0, playimage);

            IconImage chartimage = new IconImage()
            {
                Icon = IconChar.MagnifyingGlassChart,
                Foreground = Brushes.CornflowerBlue
            };
            metrics.Children.Insert(0, chartimage);

            IconImage settingsimage = new IconImage()
            {
                Icon = IconChar.Gear,
                Foreground = Brushes.CornflowerBlue
            };
            settings.Children.Insert(0, settingsimage);

            IconImage helpimage = new IconImage()
            {
                Icon = IconChar.Question,
                Foreground = Brushes.CornflowerBlue
            };
            help.Children.Insert(0, helpimage);
        }
    }
}
