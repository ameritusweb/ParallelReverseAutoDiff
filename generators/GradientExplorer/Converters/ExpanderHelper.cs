using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows;

namespace GradientExplorer.Converters
{
    public static class ExpanderHelper
    {
        public static readonly DependencyProperty AutoCollapseProperty =
            DependencyProperty.RegisterAttached(
                "AutoCollapse",
                typeof(bool),
                typeof(ExpanderHelper),
                new PropertyMetadata(false, OnAutoCollapseChanged));

        public static bool GetAutoCollapse(DependencyObject obj)
        {
            return (bool)obj.GetValue(AutoCollapseProperty);
        }

        public static void SetAutoCollapse(DependencyObject obj, bool value)
        {
            obj.SetValue(AutoCollapseProperty, value);
        }

        public static readonly DependencyProperty IsManuallyResizedProperty =
    DependencyProperty.RegisterAttached(
        "IsManuallyResized",
        typeof(bool),
        typeof(ExpanderHelper),
        new PropertyMetadata(false));

        public static bool GetIsManuallyResized(DependencyObject obj)
        {
            return (bool)obj.GetValue(IsManuallyResizedProperty);
        }

        public static void SetIsManuallyResized(DependencyObject obj, bool value)
        {
            obj.SetValue(IsManuallyResizedProperty, value);
        }

        public static readonly DependencyProperty LastHeightProperty =
        DependencyProperty.RegisterAttached(
            "LastHeight",
            typeof(double),
            typeof(ExpanderHelper),
            new PropertyMetadata(double.NaN));

        public static double GetLastHeight(DependencyObject obj)
        {
            return (double)obj.GetValue(LastHeightProperty);
        }

        public static void SetLastHeight(DependencyObject obj, double value)
        {
            obj.SetValue(LastHeightProperty, value);
        }

        private static void OnAutoCollapseChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is Expander expander)
            {
                if ((bool)e.NewValue)
                {
                    expander.Expanded += OnExpanded;
                    expander.Collapsed += OnCollapsed;
                }
                else
                {
                    expander.Expanded -= OnExpanded;
                    expander.Collapsed -= OnCollapsed;
                }
            }
        }

        private static void OnExpanded(object sender, RoutedEventArgs e)
        {
            if (ExpanderHelper.GetIsManuallyResized(sender as DependencyObject))
            {
                UpdateRowHeight(sender, GridUnitType.Pixel);
            }
            else
            {
                UpdateRowHeight(sender, GridUnitType.Star);
            }
        }

        private static void OnCollapsed(object sender, RoutedEventArgs e)
        {
            UpdateRowHeight(sender, GridUnitType.Auto);
        }

        private static void UpdateRowHeight(object sender, GridUnitType unitType)
        {
            if (sender is Expander expander && expander.Parent is Grid grid)
            {
                int row = Grid.GetRow(expander);
                var rowDefinition = grid.RowDefinitions[row];
                if (expander.Height != double.NaN)
                {
                    rowDefinition.Height = new GridLength(expander.Height, GridUnitType.Pixel);
                }
                else
                {
                    rowDefinition.Height = new GridLength(1, unitType);
                }
            }
        }
    }

}
