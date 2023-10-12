using Autofac;
using GradientExplorer.Services;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;

namespace GradientExplorer.Helpers
{
    public static class DrawDiagramBehavior
    {
        public static readonly DependencyProperty DrawDiagramProperty =
            DependencyProperty.RegisterAttached("DrawDiagram", typeof(bool), typeof(DrawDiagramBehavior), new PropertyMetadata(false, OnDrawDiagramChanged));

        public static bool GetDrawDiagram(DependencyObject obj)
        {
            return (bool)obj.GetValue(DrawDiagramProperty);
        }

        public static void SetDrawDiagram(DependencyObject obj, bool value)
        {
            obj.SetValue(DrawDiagramProperty, value);
        }

        private static void OnDrawDiagramChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is Panel panel)
            {
                if ((bool)e.NewValue)
                {
                    panel.Loaded += Panel_Loaded;
                }
                else
                {
                    panel.Loaded -= Panel_Loaded;
                    panel.UnsubscribeAll();
                }
            }
        }

        private static void Panel_Loaded(object sender, RoutedEventArgs e)
        {
            var logger = AutofacContainerProvider.Container.Resolve<ILogger>();
            if (sender is Panel panel)
            {
                logger.Log("Panel loaded.", SeverityType.Information);
                var eventAggregator = AutofacContainerProvider.Container.Resolve<IEventAggregator>();
                var addCanvasSubscription = eventAggregator.Subscribe(EventType.AddCanvasToPanel, new Action<IEventData, CancellationToken>((data, _) =>
                {
                    if (data is CanvasEventData canvasEventData)
                    {
                        panel.Children.Add(canvasEventData.Canvas);
                    }
                }), 10);
                panel.AddSubscription(addCanvasSubscription);
            }
        }
    }
}
