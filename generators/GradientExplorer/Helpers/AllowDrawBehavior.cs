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
                    canvas.UnsubscribeAll();
                }
            }
        }

        private static void Canvas_Loaded(object sender, RoutedEventArgs e)
        {
            var logger = AutofacContainerProvider.Container.Resolve<ILogger>();
            if (sender is Canvas canvas)
            {
                logger.Log("Canvas loaded.", SeverityType.Information);
                var eventAggregator = AutofacContainerProvider.Container.Resolve<IEventAggregator>();
                var addPathSubscription = eventAggregator.Subscribe(EventType.AddPathToCanvas, new Action<IEventData, CancellationToken>((data, _) =>
                {
                    if (data is PathEventData pathEventData)
                    {
                        canvas.Children.Add(pathEventData.Path);
                    }
                    eventAggregator.PostMessage(MessageType.CanvasWidth, (float)canvas.Width);
                    eventAggregator.PostMessage(MessageType.CanvasHeight, (float)canvas.Height);
                    eventAggregator.PostMessage(MessageType.CanvasActualHeight, (float)canvas.ActualHeight);
                }), 10);
                canvas.AddSubscription(addPathSubscription);

                var clearSubscription = eventAggregator.Subscribe(EventType.ClearCanvas, new Action<IEventData, CancellationToken>((data, _) =>
                {
                    canvas.Children.Clear();
                    eventAggregator.PostMessage(MessageType.CanvasWidth, (float)canvas.Width);
                    eventAggregator.PostMessage(MessageType.CanvasHeight, (float)canvas.Height);
                    eventAggregator.PostMessage(MessageType.CanvasActualHeight, (float)canvas.ActualHeight);
                }), 10);
                canvas.AddSubscription(clearSubscription);
            }
        }
    }
}
