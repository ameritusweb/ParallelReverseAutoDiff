using FontAwesome.Sharp;
using GradientExplorer.Model;
using Microsoft.VisualStudio.PlatformUI;
using System;
using System.Collections.Generic;
using System.ComponentModel;
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
    public partial class GradientToolboxControl : UserControl, INotifyPropertyChanged
    {
        public GradientToolboxControl(Version vsVersion)
        {
            InitializeComponent();

            IconImage wandimage = new IconImage()
            {
                Icon = IconChar.WandMagicSparkles,
                Foreground = Brushes.CornflowerBlue
            };
            wandimage.Height = 40;
            simplification.Children.Insert(0, wandimage);

            IconImage computationimage = new IconImage()
            {
                Icon = IconChar.DiagramProject,
                Foreground = Brushes.CornflowerBlue
            };
            computationimage.Height = 40;
            computation.Children.Insert(0, computationimage);

            IconImage playimage = new IconImage()
            {
                Icon = IconChar.Play,
                Foreground = Brushes.CornflowerBlue
            };
            playimage.Height = 40;
            debugging.Children.Insert(0, playimage);

            IconImage chartimage = new IconImage()
            {
                Icon = IconChar.MagnifyingGlassChart,
                Foreground = Brushes.CornflowerBlue
            };
            chartimage.Height = 40;
            metrics.Children.Insert(0, chartimage);

            IconImage settingsimage = new IconImage()
            {
                Icon = IconChar.Gear,
                Foreground = Brushes.CornflowerBlue
            };
            settingsimage.Height = 40;
            settings.Children.Insert(0, settingsimage);

            IconImage helpimage = new IconImage()
            {
                Icon = IconChar.Question,
                Foreground = Brushes.CornflowerBlue
            };
            helpimage.Height = 40;
            help.Children.Insert(0, helpimage);

            this.DataContext = this;

            VSColorTheme.ThemeChanged += VSColorTheme_ThemeChanged;

            var backgroundColor = VSColorTheme.GetThemedColor(EnvironmentColors.ToolWindowBackgroundColorKey);

            this.DataContext = this;
            double r = backgroundColor.R / 255.0;
            double g = backgroundColor.G / 255.0;
            double b = backgroundColor.B / 255.0;

            // Calculate the luminance
            double luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            if (luminance <= 0.5)
            {
                HoverBackground = "#FF303030";
            }
            else
            {
                HoverBackground = "#FFE1F5FE";
            }
        }

        private async void VSColorTheme_ThemeChanged(ThemeChangedEventArgs e)
        {
            var currentTheme = await ThemeManager.Instance.GetCurrentThemeAsync();
            if (currentTheme.IsDark == true)
            {
                HoverBackground = "#FF303030";
            }
            else
            {
                HoverBackground = "#FFE1F5FE";
            }
        }

        private string _hoverBackground;
        public string HoverBackground
        {
            get { return _hoverBackground; }
            set
            {
                if (_hoverBackground != value)
                {
                    _hoverBackground = value;
                    OnPropertyChanged(nameof(HoverBackground));
                }
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
