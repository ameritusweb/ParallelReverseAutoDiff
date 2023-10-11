using GradientExplorer.Commands;
using GradientExplorer.Model;
using System.ComponentModel;
using System.Windows.Input;
using System.Windows.Media;
using FontAwesome.Sharp;
using Microsoft.VisualStudio.PlatformUI;
using GradientExplorer.Helpers;

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

        // ICommand for clicking on "Gradient Toolbox" breadcrumb
        public ICommand GoToUniformGridCommand { get; }

        public GradientToolboxViewModel()
        {
            SimplificationViewCommand = new AsyncRelayCommand(SimplificationViewAsync, CanSimplificationView);
            ComputationViewCommand = new AsyncRelayCommand(ComputationViewAsync, CanComputationView);
            DebuggingViewCommand = new AsyncRelayCommand(DebuggingViewAsync, CanDebuggingView);
            MetricsViewCommand = new AsyncRelayCommand(MetricsViewAsync, CanMetricsView);
            SettingsViewCommand = new AsyncRelayCommand(SettingsViewAsync, CanSettingsView);
            HelpViewCommand = new AsyncRelayCommand(HelpViewAsync, CanHelpView);
            GoToUniformGridCommand = new RelayCommand(UniformGridView, CanUniformGridView);
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

        private GradientToolView _currentView;

        public GradientToolView CurrentView
        {
            get { return _currentView; }
            set
            {
                if (_currentView != value)
                {
                    _currentView = value;
                    OnPropertyChanged(nameof(CurrentView));
                    OnPropertyChanged(nameof(ToolName));  // Notify that ToolName has also changed
                }
            }
        }

        public IconImageViewModel SimplificationIcon { get; set; }

        public IconImageViewModel ComputationIcon { get; set; }

        public IconImageViewModel DebuggingIcon { get; set; }

        public IconImageViewModel MetricsIcon { get; set; }

        public IconImageViewModel SettingsIcon { get; set; }

        public IconImageViewModel HelpIcon { get; set; }

        public string ToolName
        {
            get
            {
                switch (CurrentView)
                {
                    case GradientToolView.UniformGrid: return "Uniform Grid";
                    case GradientToolView.SimplificationTool: return "Simplification Tool";
                    case GradientToolView.ComputationTool: return "Computation Tool";
                    case GradientToolView.DebuggingTool: return "Debugging Tool";
                    case GradientToolView.MetricsTool: return "Metrics Tool";
                    case GradientToolView.SettingsTool: return "Settings Tool";
                    case GradientToolView.HelpTool: return "Help Tool";
                    // ... other cases
                    default: return "Unknown Tool";
                }
            }
        }

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

        private void UniformGridView()
        {
            this.CurrentView = GradientToolView.UniformGrid;
        }

        private bool CanUniformGridView()
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
