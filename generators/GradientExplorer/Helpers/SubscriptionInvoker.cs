using GradientExplorer.Services;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace GradientExplorer.Helpers
{
    public class SubscriptionInvoker<T> : ISubscriptionInvoker<T> where T : IEventData
    {
        private readonly ILogger _logger;

        public SubscriptionInvoker(ILogger logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        public void InvokeSubscribers(
            EventType eventType,
            ConcurrentDictionary<int, ThreadSafeList<SubscriptionBase>> subscribers,
            T eventData)
        {
            subscribers
                .OrderByDescending(kvp => kvp.Key)
                .ToList()
                .ForEach(kvp => InvokePriorityGroup(kvp.Key, kvp.Value, eventType, eventData));
        }

        private void InvokePriorityGroup(
            int priority,
            ThreadSafeList<SubscriptionBase> subscriptionList,
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
                subscription.CancellationTokenSource.Token.ThrowIfCancellationRequested();
                subscription.Action(eventData, subscription.CancellationTokenSource.Token);
            }
            catch (OperationCanceledException)
            {
                _logger.Log(nameof(SubscriptionInvoker<T>), $"Cancellation requested for event type {eventType}", SeverityType.Warning);
            }
            catch (Exception ex)
            {
                _logger.Log(nameof(SubscriptionInvoker<T>), $"Encountered an exception while invoking a subscriber for event type {eventType}: {ex.Message}", SeverityType.Error);
            }
            finally
            {
                subscription.Stopwatch.Stop();
            }
        }

        public async Task InvokeSubscribersAsync(
            EventType eventType,
            ConcurrentDictionary<int, ThreadSafeList<SubscriptionBase>> asyncSubscribers,
            T eventData)
        {
            var tasks = asyncSubscribers
                .OrderByDescending(kvp => kvp.Key)
                .SelectMany(kvp => InvokeAsyncPriorityGroup(kvp.Key, kvp.Value, eventType, eventData))
                .ToArray();

            await Task.WhenAll(tasks).ConfigureAwait(false);
        }

        private IEnumerable<Task> InvokeAsyncPriorityGroup(
            int priority,
            ThreadSafeList<SubscriptionBase> subscriptionList,
            EventType eventType,
            T eventData)
        {
            return subscriptionList
                .OfType<SubscriptionAsync<T>>()
                .Where(subscription => !subscription.CancellationTokenSource.Token.IsCancellationRequested &&
                                       (subscription.Filter == null || subscription.Filter(eventData)))
                .Select(subscription => InvokeAsyncSubscriptionAction(subscription, eventType, eventData));
        }

        private async Task InvokeAsyncSubscriptionAction(
            SubscriptionAsync<T> subscription,
            EventType eventType,
            T eventData)
        {
            subscription.Stopwatch.Start();

            try
            {
                subscription.CancellationTokenSource.Token.ThrowIfCancellationRequested();
                await subscription.AsyncAction(eventData, subscription.CancellationTokenSource.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                _logger.Log(nameof(SubscriptionInvoker<T>), $"Cancellation requested for event type {eventType}", SeverityType.Warning);
            }
            catch (Exception ex)
            {
                _logger.Log(nameof(SubscriptionInvoker<T>), $"Encountered an exception while invoking an async subscriber for event type {eventType}: {ex.Message}", SeverityType.Error);
            }
            finally
            {
                subscription.Stopwatch.Stop();
            }
        }
    }
}
