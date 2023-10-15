using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace GradientExplorer.Controls
{
    public class SortableListBox : ListBox
    {
        // Define the DependencyProperty for IsPotentialDrag
        public static readonly DependencyProperty IsPotentialDragProperty =
            DependencyProperty.Register("IsPotentialDrag", typeof(bool), typeof(SortableListBox), new PropertyMetadata(false));

        public bool IsPotentialDrag
        {
            get => (bool)GetValue(IsPotentialDragProperty);
            set => SetValue(IsPotentialDragProperty, value);
        }

        // Define the DependencyProperty for IsAnimating
        public static readonly DependencyProperty IsAnimatingProperty =
            DependencyProperty.Register("IsAnimating", typeof(bool), typeof(SortableListBox), new PropertyMetadata(false));

        public bool IsAnimating
        {
            get => (bool)GetValue(IsAnimatingProperty);
            set => SetValue(IsAnimatingProperty, value);
        }
    }
}
