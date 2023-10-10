using GradientExplorer.Commands;
using GradientExplorer.Model;
using System.ComponentModel;
using System.Windows.Input;
using System.Windows.Media;
using FontAwesome.Sharp;
using Microsoft.VisualStudio.PlatformUI;

namespace GradientExplorer.ViewModels
{
    public class GradientToolboxViewModel : INotifyPropertyChanged
    {

        public ICommand SimplificationViewCommand { get; }
        public ICommand ComputationViewCommand { get; }
        public ICommand DebuggingViewCommand { get; }
        public ICommand MetricsViewCommand { get; }
        public ICommand SettingsViewCommand { get; }
        public ICommand HelpViewCommand { get; }


        public GradientToolboxViewModel()
        {
            SimplificationViewCommand = new AsyncRelayCommand(SimplificationViewAsync, CanSimplificationView);
            ComputationViewCommand = new AsyncRelayCommand(ComputationViewAsync, CanComputationView);
            DebuggingViewCommand = new AsyncRelayCommand(DebuggingViewAsync, CanDebuggingView);
            MetricsViewCommand = new AsyncRelayCommand(MetricsViewAsync, CanMetricsView);
            SettingsViewCommand = new AsyncRelayCommand(SettingsViewAsync, CanSettingsView);
            HelpViewCommand = new AsyncRelayCommand(HelpViewAsync, CanHelpView);
            SimplificationIcon = new IconImageViewModel
            {
                Icon = IconChar.WandMagicSparkles,
                Foreground = Brushes.CornflowerBlue,
                Height = 40,
            };
            ComputationIcon = new IconImageViewModel
            {
                Icon = IconChar.DiagramProject,
                Foreground = Brushes.CornflowerBlue,
                Height = 40,
            };
            DebuggingIcon = new IconImageViewModel
            {
                Icon = IconChar.Play,
                Foreground = Brushes.CornflowerBlue,
                Height = 40,
            };
            MetricsIcon = new IconImageViewModel
            {
                Icon = IconChar.MagnifyingGlassChart,
                Foreground = Brushes.CornflowerBlue,
                Height = 40,
            };
            SettingsIcon = new IconImageViewModel
            {
                Icon = IconChar.Gear,
                Foreground = Brushes.CornflowerBlue,
                Height = 40,
            };
            HelpIcon = new IconImageViewModel
            {
                Icon = IconChar.Question,
                Foreground = Brushes.CornflowerBlue,
                Height = 40,
            };
        }

        public IconImageViewModel SimplificationIcon { get; set; }

        public IconImageViewModel ComputationIcon { get; set; }

        public IconImageViewModel DebuggingIcon { get; set; }

        public IconImageViewModel MetricsIcon { get; set; }

        public IconImageViewModel SettingsIcon { get; set; }

        public IconImageViewModel HelpIcon { get; set; }

        private async Task SimplificationViewAsync()
        {
            var docView = await VS.Documents.GetActiveDocumentViewAsync();
            if (docView != null)
            {

            }
        }

        private bool CanSimplificationView()
        {
            // Validation logic to enable/disable button
            return true;
        }

        private async Task ComputationViewAsync()
        {
            var docView = await VS.Documents.GetActiveDocumentViewAsync();
            if (docView != null)
            {

            }
        }

        private bool CanComputationView()
        {
            // Validation logic to enable/disable button
            return true;
        }

        private async Task DebuggingViewAsync()
        {
            var docView = await VS.Documents.GetActiveDocumentViewAsync();
            if (docView != null)
            {

            }
        }

        private bool CanDebuggingView()
        {
            // Validation logic to enable/disable button
            return true;
        }

        private async Task MetricsViewAsync()
        {
            var docView = await VS.Documents.GetActiveDocumentViewAsync();
            if (docView != null)
            {

            }
        }

        private bool CanMetricsView()
        {
            // Validation logic to enable/disable button
            return true;
        }

        private async Task SettingsViewAsync()
        {
            var docView = await VS.Documents.GetActiveDocumentViewAsync();
            if (docView != null)
            {

            }
        }

        private bool CanSettingsView()
        {
            // Validation logic to enable/disable button
            return true;
        }

        private async Task HelpViewAsync()
        {
            var docView = await VS.Documents.GetActiveDocumentViewAsync();
            if (docView != null)
            {

            }
        }

        private bool CanHelpView()
        {
            // Validation logic to enable/disable button
            return true;
        }

        public void LoadData()
        {
            VSColorTheme.ThemeChanged += VSColorTheme_ThemeChanged;

            var backgroundColor = VSColorTheme.GetThemedColor(EnvironmentColors.ToolWindowBackgroundColorKey);

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
