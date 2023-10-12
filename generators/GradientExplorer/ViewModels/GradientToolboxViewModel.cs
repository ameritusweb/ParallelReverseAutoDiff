using GradientExplorer.Commands;
using GradientExplorer.Model;
using System.ComponentModel;
using System.Windows.Input;
using System.Windows.Media;
using FontAwesome.Sharp;
using Microsoft.VisualStudio.PlatformUI;
using GradientExplorer.Helpers;
using GradientExplorer.Services;

namespace GradientExplorer.ViewModels
{
    public class GradientToolboxViewModel : INotifyPropertyChanged
    {
        private ILogger logger;
        public ICommand SimplificationViewCommand { get; }
        public ICommand ComputationViewCommand { get; }
        public ICommand DebuggingViewCommand { get; }
        public ICommand MetricsViewCommand { get; }
        public ICommand SettingsViewCommand { get; }
        public ICommand HelpViewCommand { get; }

        // ICommand for clicking on "Gradient Toolbox" breadcrumb
        public ICommand GoToUniformGridCommand { get; }

        public GradientToolboxViewModel(ILogger logger)
        {
            this.logger = logger;
            SimplificationViewCommand = new RelayCommand(SimplificationView, CanSimplificationView);
            ComputationViewCommand = new RelayCommand(ComputationView, CanComputationView);
            DebuggingViewCommand = new RelayCommand(DebuggingView, CanDebuggingView);
            MetricsViewCommand = new RelayCommand(MetricsView, CanMetricsView);
            SettingsViewCommand = new RelayCommand(SettingsView, CanSettingsView);
            HelpViewCommand = new RelayCommand(HelpView, CanHelpView);
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

            logger.Log("Gradient Toolbox started.", SeverityType.Information);
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
                    case GradientToolView.MetricsTool: return "Metrics Display";
                    case GradientToolView.SettingsTool: return "Settings";
                    case GradientToolView.HelpTool: return "Help";
                    // ... other cases
                    default: return "Unknown Tool";
                }
            }
        }

        private void SimplificationView()
        {
            CurrentView = GradientToolView.SimplificationTool;
        }

        private bool CanSimplificationView()
        {
            // Validation logic to enable/disable button
            return true;
        }

        private void ComputationView()
        {
            CurrentView = GradientToolView.ComputationTool;
        }

        private bool CanComputationView()
        {
            // Validation logic to enable/disable button
            return true;
        }

        private void DebuggingView()
        {
            CurrentView = GradientToolView.DebuggingTool;
        }

        private bool CanDebuggingView()
        {
            // Validation logic to enable/disable button
            return true;
        }

        private void MetricsView()
        {
            CurrentView = GradientToolView.MetricsTool;
        }

        private bool CanMetricsView()
        {
            // Validation logic to enable/disable button
            return true;
        }

        private void SettingsView()
        {
            CurrentView = GradientToolView.SettingsTool;
        }

        private bool CanSettingsView()
        {
            // Validation logic to enable/disable button
            return true;
        }

        private void HelpView()
        {
            CurrentView = GradientToolView.HelpTool;
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
                ForegroundColor = "#FFFFFFFF";
            }
            else
            {
                HoverBackground = "#FFE1F5FE";
                ForegroundColor = "#FF000000";
            }
        }

        private async void VSColorTheme_ThemeChanged(ThemeChangedEventArgs e)
        {
            var currentTheme = await ThemeManager.Instance.GetCurrentThemeAsync();
            if (currentTheme.IsDark == true)
            {
                HoverBackground = "#FF303030";
                ForegroundColor = "#FFFFFFFF";
            }
            else
            {
                HoverBackground = "#FFE1F5FE";
                ForegroundColor = "#FF000000";
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

        private string _foregroundColor;
        public string ForegroundColor
        {
            get { return _foregroundColor; }
            set
            {
                if (_foregroundColor != value)
                {
                    _foregroundColor = value;
                    OnPropertyChanged(nameof(ForegroundColor));
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
