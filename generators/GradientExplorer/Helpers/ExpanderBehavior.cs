using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows;
using System.Windows.Controls.Primitives;

namespace GradientExplorer.Helpers
{
    public static class ExpanderBehavior
    {
        public static readonly DependencyProperty TrackHeightProperty =
            DependencyProperty.RegisterAttached("TrackHeight", typeof(bool), typeof(ExpanderBehavior), new PropertyMetadata(false, OnTrackHeightChanged));

        private static readonly Dictionary<Expander, double?> LastHeights = new Dictionary<Expander, double?>();

        public static bool GetTrackHeight(DependencyObject obj)
        {
            return (bool)obj.GetValue(TrackHeightProperty);
        }

        public static void SetTrackHeight(DependencyObject obj, bool value)
        {
            obj.SetValue(TrackHeightProperty, value);
        }

        private static void OnTrackHeightChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is Expander expander)
            {
                if ((bool)e.NewValue)
                {
                    expander.Expanded += Expander_Expanded;
                    expander.Collapsed += Expander_Collapsed;
                }
                else
                {
                    expander.Expanded -= Expander_Expanded;
                    expander.Collapsed -= Expander_Collapsed;
                    LastHeights.Remove(expander);
                }
            }
        }

        private static void Expander_Expanded(object sender, RoutedEventArgs e)
        {
            if (sender is Expander expander)
            {
                if (LastHeights.TryGetValue(expander, out var lastHeight))
                {
                    expander.Height = lastHeight ?? double.NaN;
                }
            }
        }

        private static void Expander_Collapsed(object sender, RoutedEventArgs e)
        {
            if (sender is Expander expander)
            {
                LastHeights[expander] = expander.ActualHeight;
                expander.Height = double.NaN; // Reset to auto-size
            }
        }

        public static readonly DependencyProperty EnableDragResizeProperty =
        DependencyProperty.RegisterAttached("EnableDragResize", typeof(bool), typeof(ExpanderBehavior), new PropertyMetadata(false, OnEnableDragResizeChanged));

        public static bool GetEnableDragResize(DependencyObject obj)
        {
            return (bool)obj.GetValue(EnableDragResizeProperty);
        }

        public static void SetEnableDragResize(DependencyObject obj, bool value)
        {
            obj.SetValue(EnableDragResizeProperty, value);
        }

        private static void OnEnableDragResizeChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is Thumb thumb)
            {
                if ((bool)e.NewValue)
                {
                    thumb.DragDelta += Resizer_DragDelta;
                    thumb.DragCompleted += Resizer_DragCompleted;
                }
                else
                {
                    thumb.DragDelta -= Resizer_DragDelta;
                    thumb.DragCompleted -= Resizer_DragCompleted;
                }
            }
        }

        private static void Resizer_DragDelta(object sender, DragDeltaEventArgs e)
        {
            if (sender is Thumb thumb && thumb.DataContext is Expander expander)
            {
                double newHeight = expander.ActualHeight + e.VerticalChange;
                newHeight = Math.Max(newHeight, 50);  // Set a minimum height
                expander.Height = newHeight;
                LastHeights[expander] = newHeight;
            }
        }

        private static void Resizer_DragCompleted(object sender, DragCompletedEventArgs e)
        {
            // No need to reset the height here as it's managed by the other events.
        }
    }
}
