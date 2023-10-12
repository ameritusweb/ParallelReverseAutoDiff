using System;
using System.Collections.Concurrent;

namespace GradientExplorer.Helpers
{
    public interface ISubscriptionInvoker<T> where T : IEventData
    {
        void InvokeSubscribers(
            EventType eventType,
            ConcurrentDictionary<int, ThreadSafeList<SubscriptionBase>> subscribers,
            T eventData);

        Task InvokeSubscribersAsync(
            EventType eventType,
            ConcurrentDictionary<int, ThreadSafeList<SubscriptionBase>> asyncSubscribers,
            T eventData);
    }
}
