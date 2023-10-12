using GradientExplorer.Services;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace GradientExplorer.Helpers
{
    public class AsyncSubscriptionInvoker<T> where T : IEventData
    {
        private readonly ILogger _logger;

        public AsyncSubscriptionInvoker(ILogger logger)
        {
            _logger = logger;
        }

        public async Task InvokeSubscribersAsync(EventType eventType, ConcurrentDictionary<int, List<SubscriptionBase>> asyncSubscribers, T eventData)
        {
            foreach (var (priority, subscriptionList) in asyncSubscribers.OrderByDescending(kvp => kvp.Key))
            {
                await InvokeAsyncPriorityGroup(priority, subscriptionList, eventType, eventData).ConfigureAwait(false);
            }
        }

        private async Task InvokeAsyncPriorityGroup(
            int priority,
            List<SubscriptionBase> subscriptionList,
            EventType eventType,
            T eventData)
        {
            var tasks = subscriptionList
                .OfType<SubscriptionAsync<T>>()
                .Where(subscription => !subscription.CancellationTokenSource.Token.IsCancellationRequested &&
                                       (subscription.Filter == null || subscription.Filter(eventData)))
                .Select(subscription => InvokeAsyncSubscriptionAction(subscription, eventType, eventData))
                .ToArray();

            await Task.WhenAll(tasks).ConfigureAwait(false);
        }

        private async Task InvokeAsyncSubscriptionAction(
            SubscriptionAsync<T> subscription,
            EventType eventType,
            T eventData)
        {
            subscription.Stopwatch.Start();

            try
            {
                await subscription.AsyncAction(eventData, subscription.CancellationTokenSource.Token).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.Log($"Encountered an exception while invoking an async subscriber for event type {eventType}: {ex.Message}", SeverityType.Error);
            }
            finally
            {
                subscription.Stopwatch.Stop();
            }
        }
    }
}
