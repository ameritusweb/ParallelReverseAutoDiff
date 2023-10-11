using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows;

namespace GradientExplorer.Helpers
{
    public static class FrameworkElementExtensions
    {
        public static void AddSubscription(this FrameworkElement element, ISubscriptionBase subscription)
        {
            if (element.Tag == null)
            {
                element.Tag = new SubscriptionQueue<ISubscriptionBase>();
            }
            if (element.Tag is SubscriptionQueue<ISubscriptionBase> queue)
            {
                queue.Enqueue(subscription);
            }
        }

        public static U GetQueue<U>(this FrameworkElement element) where U : ConcurrentQueue<ISubscriptionBase>
        {
            var tag = element.Tag;
            if (tag != null)
            {
                var tagType = tag.GetType();
                var targetType = typeof(U);

                if (tagType.IsGenericType && targetType.IsGenericType)
                {
                    var tagBase = tagType.GetGenericTypeDefinition();
                    var targetBase = targetType.GetGenericTypeDefinition();

                    if (tagBase == targetBase)
                    {
                        var tagArg = tagType.GetGenericArguments()[0];
                        var targetArg = targetType.GetGenericArguments()[0];

                        if (targetArg.IsAssignableFrom(tagArg))
                        {
                            return (U)tag;
                        }
                    }
                }
            }
            return default(U);
        }

        public static void UnsubscribeAll(this FrameworkElement canvas)
        {
            if (canvas.Tag is SubscriptionQueue<ISubscriptionBase> queue)
            {
                while (queue.Count > 0)
                {
                    var result = queue.TryDequeue(out ISubscriptionBase subscription);
                    if (result)
                    {
                        subscription.Dispose();
                    }
                }
            }
        }
    }
}
