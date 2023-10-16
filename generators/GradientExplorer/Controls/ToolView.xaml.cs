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

        private static void OnInnerContentChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            var toolView = (ToolView)d;
            toolView.InnerContentGrid.Children.Clear();
            if (e.NewValue is FrameworkElement element)
            {
                toolView.InnerContentGrid.Children.Add(element);
            }
        }
    }
}
