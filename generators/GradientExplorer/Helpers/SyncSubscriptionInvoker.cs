using GradientExplorer.Services;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Helpers
{
    public class SyncSubscriptionInvoker<T> where T : IEventData
    {
        private readonly ILogger _logger;

        public SyncSubscriptionInvoker(ILogger logger)
        {
            _logger = logger;
        }

        public void InvokeSubscribers(
            EventType eventType,
            ConcurrentDictionary<int, List<SubscriptionBase>> subscribers,
            T eventData)
        {
            subscribers
                .OrderByDescending(kvp => kvp.Key)
                .ToList()
                .ForEach(kvp => InvokePriorityGroup(kvp.Key, kvp.Value, eventType, eventData));
        }

        private void InvokePriorityGroup(
            int priority,
            List<SubscriptionBase> subscriptionList,
            EventType eventType,
            T eventData)
        {
            subscriptionList
                .OfType<Subscription<T>>()
                .Where(subscription => !subscription.CancellationTokenSource.Token.IsCancellationRequested &&
                                       (subscription.Filter == null || subscription.Filter(eventData)))
                .ToList()
                .ForEach(subscription => InvokeSubscriptionAction(subscription, eventType, eventData));
        }

        private void InvokeSubscriptionAction(Subscription<T> subscription, EventType eventType, T eventData)
        {
            subscription.Stopwatch.Start();

            try
            {
                subscription.Action(eventData, subscription.CancellationTokenSource.Token);
            }
            catch (Exception ex)
            {
                _logger.Log($"Encountered an exception while invoking a subscriber for event type {eventType}: {ex.Message}", SeverityType.Error);
            }
            finally
            {
                subscription.Stopwatch.Stop();
            }
        }
    }

}
