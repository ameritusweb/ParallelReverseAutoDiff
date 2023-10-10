using Autofac;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;

namespace GradientExplorer.Helpers
{
    public static class AllowDrawBehavior
    {
        public static readonly DependencyProperty AllowDrawProperty =
            DependencyProperty.RegisterAttached("AllowDraw", typeof(bool), typeof(AllowDrawBehavior), new PropertyMetadata(false, OnAllowDrawChanged));

        public static bool GetAllowDraw(DependencyObject obj)
        {
            return (bool)obj.GetValue(AllowDrawProperty);
        }

        public static void SetAllowDraw(DependencyObject obj, bool value)
        {
            obj.SetValue(AllowDrawProperty, value);
        }

        private static void OnAllowDrawChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is Canvas canvas)
            {
                if ((bool)e.NewValue)
                {
                    canvas.Loaded += Canvas_Loaded;
                }
                else
                {
                    canvas.Loaded -= Canvas_Loaded;
                }
            }
        }

        private static void Canvas_Loaded(object sender, RoutedEventArgs e)
        {
            if (sender is Canvas canvas)
            {
                var eventAggregator = AutofacContainerProvider.Container.Resolve<IEventAggregator>();
                eventAggregator.Subscribe(EventType.AddPathToCanvas, new Action<IEventData>(data =>
                {
                    if (data is AddPathToCanvasEventData addPathToCanvasEventData)
                    {
                        canvas.Children.Add(addPathToCanvasEventData.Path);
                    }
                }));
            }
        }
    }
}
