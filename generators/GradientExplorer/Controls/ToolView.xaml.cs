using GradientExplorer.Helpers;
using System.Windows;
using System.Windows.Controls;

namespace GradientExplorer.Controls
{
    /// <summary>
    /// Interaction logic for ToolView.xaml
    /// </summary>
    public partial class ToolView : UserControl
    {
        public ToolView()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty InnerContentProperty =
            DependencyProperty.Register("InnerContent", typeof(FrameworkElement), typeof(ToolView),
                new PropertyMetadata(null, OnInnerContentChanged));

        public object InnerContent
        {
            get { return GetValue(InnerContentProperty); }
            set { SetValue(InnerContentProperty, value); }
        }

        public static readonly DependencyProperty InnerContentDataContextProperty =
            DependencyProperty.Register("InnerContentDataContext", typeof(IViewModel), typeof(ToolView),
                new PropertyMetadata(null));

        public object InnerContentDataContext
        {
            get { return GetValue(InnerContentDataContextProperty); }
            set { SetValue(InnerContentDataContextProperty, value); }
        }

        private static void OnInnerContentChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            var toolView = (ToolView)d;
            toolView.InnerContentGrid.Children.Clear();
            if (e.NewValue is FrameworkElement element)
            {
                toolView.InnerContentGrid.Children.Add(element);
            }
        }

        private void InnerContentGrid_Loaded(object sender, RoutedEventArgs e)
        {
            var innerContentGrid = (Grid)sender;
            innerContentGrid.DataContext = InnerContentDataContext;
        }
    }
}
