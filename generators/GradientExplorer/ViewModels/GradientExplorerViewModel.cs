﻿using GradientExplorer.Commands;
using GradientExplorer.Diagram;
using GradientExplorer.LaTeX.Wpf;
using GradientExplorer.Model;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Text;
using System.ComponentModel;
using System.Linq;
using System.Windows.Input;
using System.Windows.Media;
using Microsoft.CodeAnalysis;
using FontAwesome.Sharp;
using Microsoft.VisualStudio.PlatformUI;
using GradientExplorer.Extensions;
using GradientExplorer.Helpers;
using GradientExplorer.Services;

namespace GradientExplorer.ViewModels
{
    public class GradientExplorerViewModel : INotifyPropertyChanged
    {
        private DiagramCanvas currentDiagram;
        private WpfMathPainter painter;
        private IGradientGraphFactory gradientGraphFactory;
        private IEventAggregator eventAggregator;
        private IMessageRetriever messageRetriever;
        private ILaTeXBuilder laTeXBuilder;
        private ILogger logger;

        public ICommand ComputeGradientCommand { get; }

        public GradientExplorerViewModel(IGradientGraphFactory gradientGraphFactory, IEventAggregator eventAggregator, IMessageRetriever messageRetriever, ILaTeXBuilder laTeXBuilder, ILogger logger)
        {
            this.gradientGraphFactory = gradientGraphFactory;
            this.eventAggregator = eventAggregator;
            this.messageRetriever = messageRetriever;
            this.laTeXBuilder = laTeXBuilder;
            this.logger = logger;
            ComputeGradientCommand = new AsyncRelayCommand(ComputeGradientAsync, CanComputeGradient);
            GradientTabIcon = new IconImageViewModel
            {
                Icon = IconChar.Play,
                Foreground = Brushes.CornflowerBlue,
                Height = 20,
                Transforms = new TransformGroup
                {
                    Children = new TransformCollection
                    {
                        new RotateTransform(90),
                        new TranslateTransform(18, 2.5)
                    }
                },
            };
            ComputationTabIcon = new IconImageViewModel
            {
                Icon = IconChar.DiagramProject,
                Foreground = Brushes.CornflowerBlue,
                Height = 20,
            };

            ComputationTabViewModel = new ComputationTabViewModel(this, logger);

            logger.Log(nameof(GradientExplorerViewModel), "Gradient Explorer started", SeverityType.Information);
        }

        public Version VSVersion { get; set; }

        public ComputationTabViewModel ComputationTabViewModel { get; set; }

        public IconImageViewModel GradientTabIcon { get; set; }

        public IconImageViewModel ComputationTabIcon { get; set; }

        private string _headline;

        public string Headline
        {
            get => _headline;
            set
            {
                if (_headline != value)
                {
                    _headline = value;
                    OnPropertyChanged(nameof(Headline));
                }
            }
        }

        private SolidColorBrush _backgroundColor;

        public SolidColorBrush BackgroundColor
        {
            get => _backgroundColor;
            set
            {
                if (_backgroundColor != value)
                {
                    _backgroundColor = value;
                    OnPropertyChanged(nameof(BackgroundColor));
                }
            }
        }

        public void UpdateBackgroundColor()
        {
            var backgroundColor = VSColorTheme.GetThemedColor(EnvironmentColors.ToolWindowBackgroundColorKey);
            BackgroundColor = new SolidColorBrush(System.Windows.Media.Color.FromArgb(backgroundColor.A, backgroundColor.R, backgroundColor.G, backgroundColor.B));
            double r = backgroundColor.R / 255.0;
            double g = backgroundColor.G / 255.0;
            double b = backgroundColor.B / 255.0;

            // Calculate the luminance
            double luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            if (luminance <= 0.5)
            {
                IsDarkMode = true;
            }
            else
            {
                IsDarkMode = false;
            }
        }

        public void LoadData()
        {
            Headline = $"Visual Studio v{VSVersion}";

            VSColorTheme.ThemeChanged += VSColorTheme_ThemeChanged;

            UpdateDarkModeProperties();
            
        }

        private string _expanderForeground;
        public string ExpanderForeground
        {
            get { return _expanderForeground; }
            set
            {
                if (_expanderForeground != value)
                {
                    _expanderForeground = value;
                    OnPropertyChanged(nameof(ExpanderForeground));
                }
            }
        }

        private bool _isDarkMode;
        public bool IsDarkMode
        {
            get { return _isDarkMode; }
            set
            {
                if (_isDarkMode != value)
                {
                    _isDarkMode = value;
                    UpdateDarkModeProperties();
                    OnPropertyChanged(nameof(IsDarkMode));
                }
            }
        }

        private void UpdateDarkModeProperties()
        {
            ExpanderForeground = IsDarkMode ? Brushes.White.ToHex() : Brushes.Black.ToHex();
        }

        private async void VSColorTheme_ThemeChanged(ThemeChangedEventArgs e)
        {
            var backgroundColor = VSColorTheme.GetThemedColor(EnvironmentColors.ToolWindowBackgroundColorKey);
            BackgroundColor = new SolidColorBrush(System.Windows.Media.Color.FromArgb(backgroundColor.A, backgroundColor.R, backgroundColor.G, backgroundColor.B));

            var currentTheme = await ThemeManager.Instance.GetCurrentThemeAsync();
            IsDarkMode = currentTheme.IsDark;
            if (this.currentDiagram != null)
            {
                bool changed = this.currentDiagram.UpdateTheme(currentTheme);
                if (changed && painter != null)
                {
                    eventAggregator.PublishAsync(EventType.ClearCanvas, new ClearEventData()).Wait();
                    var wpfCanvas = new WpfCanvas(eventAggregator, messageRetriever);
                    painter.Draw(wpfCanvas);
                }
            }
        }

        private async Task ComputeGradientAsync()
        {
            var docView = await VS.Documents.GetActiveDocumentViewAsync();
            if (docView != null)
            {
                var snapshot = docView.TextView.TextSnapshot;
                var text = string.Join("\n", snapshot.Lines.Select(x => x.GetText()));

                // Convert the text into a SourceText object for Roslyn to understand
                var sourceText = SourceText.From(text);

                // Parse the SourceText into a SyntaxTree
                var syntaxTree = CSharpSyntaxTree.ParseText(sourceText);

                var root = syntaxTree.GetRoot();
                var methods = root.DescendantNodes().OfType<MethodDeclarationSyntax>();
                var forwardMethod = methods.FirstOrDefault(m => m.Identifier.Text == "Forward");
                if (forwardMethod != null)
                {
                    var gradientGraph = await gradientGraphFactory.CreateGradientGraphAsync(forwardMethod);
                    eventAggregator.PublishAsync(EventType.ClearCanvas, new ClearEventData()).Wait();
                    var wpfCanvas = new WpfCanvas(eventAggregator, messageRetriever);
                    painter = new WpfMathPainter();
                    painter.LaTeX = gradientGraph.ToLaTeX(laTeXBuilder);
                    Headline = painter.LaTeX;
                    painter.Draw(wpfCanvas);

                    var currentTheme = await ThemeManager.Instance.GetCurrentThemeAsync();
                    if (currentDiagram == null)
                    {
                        currentDiagram = new DiagramCanvas(eventAggregator, gradientGraph.DeepCopy(), currentTheme);
                        currentDiagram.BuildGraph();
                        ScaleTransform flipTransform = new ScaleTransform(1, -1);
                        eventAggregator.PublishAsync(EventType.SetPanelLayoutTransform, new PanelLayoutTransformEventData { LayoutTransform = flipTransform }).Wait();
                        currentDiagram.AddToPanel();
                    }
                    else
                    {
                        currentDiagram.Reinitialize(gradientGraph.DeepCopy(), currentTheme);
                        currentDiagram.BuildGraph();
                    }
                }
            }
        }

        private bool CanComputeGradient()
        {
            // Validation logic to enable/disable button
            return true;
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }

}
