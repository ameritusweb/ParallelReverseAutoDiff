using GradientExplorer.Helpers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;

namespace GradientExplorer.Services
{
    public class EventAggregator : IEventAggregator
    {
        private readonly ConcurrentDictionary<EventType, ConcurrentDictionary<int, List<SubscriptionBase>>> _syncSubscriptions;
        private readonly ConcurrentDictionary<EventType, ConcurrentDictionary<int, List<SubscriptionBase>>> _asyncSubscriptions;
        private readonly ConcurrentDictionary<MessageType, UniqueTypeSet> _messages = new ConcurrentDictionary<MessageType, UniqueTypeSet>();
        private readonly ILogger _logger;

        public EventAggregator(
            ILogger logger,
            ConcurrentDictionary<EventType, ConcurrentDictionary<int, List<SubscriptionBase>>> syncSubscriptions,
            ConcurrentDictionary<EventType, ConcurrentDictionary<int, List<SubscriptionBase>>> asyncSubscriptions)
        {
            _logger = logger;
            this._syncSubscriptions = syncSubscriptions;
            this._asyncSubscriptions = asyncSubscriptions;
        }

        public Subscription<T> Subscribe<T>(EventType eventType, Action<T, CancellationToken> action, int priority, Func<T, bool> filter = null) where T : IEventData
        {
            var subscribers = _syncSubscriptions.GetOrAdd(eventType, _ => new ConcurrentDictionary<int, List<SubscriptionBase>>());

            var subscription = new Subscription<T>(action, priority, filter, subscribers);
            if (!subscribers.ContainsKey(priority))
            {
                subscribers[priority] = new List<SubscriptionBase>();
            }

            subscribers[priority].Add(subscription);

            return subscription;
        }

        public SubscriptionAsync<T> SubscribeAsync<T>(EventType eventType, Func<T, CancellationToken, Task> asyncAction, int priority, Func<T, bool> filter = null) where T : IEventData
        {
            var asyncSubscribers = _asyncSubscriptions.GetOrAdd(eventType, _ => new ConcurrentDictionary<int, List<SubscriptionBase>>());

            var subscription = new SubscriptionAsync<T>(asyncAction, priority, filter, asyncSubscribers);
            if (!asyncSubscribers.ContainsKey(priority))
            {
                asyncSubscribers[priority] = new List<SubscriptionBase>();
            }

            asyncSubscribers[priority].Add(subscription);

            return subscription;
        }

        public async Task PublishAsync<T>(EventType eventType, T eventData) where T : IEventData
        {
            _logger.Log($"Message published with event type [{eventType}]", SeverityType.Information);

            bool foundSubscribers = false;

            // Handle synchronous subscribers
            if (_syncSubscriptions.TryGetValue(eventType, out var syncSubscribers))
            {
                foundSubscribers = true;
                var invoker = new SyncSubscriptionInvoker<T>(_logger);
                invoker.InvokeSubscribers(eventType, syncSubscribers, eventData);

            }

            // Handle asynchronous subscribers
            if (_asyncSubscriptions.TryGetValue(eventType, out var asyncSubscribers))
            {
                foundSubscribers = true;
                var invoker = new AsyncSubscriptionInvoker<T>(_logger);
                await invoker.InvokeSubscribersAsync(eventType, asyncSubscribers, eventData);
            }

            if (!foundSubscribers)
            {
                throw new InvalidOperationException($"No subscribers found for event type [{eventType}].");
            }
        }

        public void PostMessage<T>(MessageType messageType, T message)
        {
            _logger.Log($"Message posted with message type [{messageType}] and data type {typeof(T).Name}", SeverityType.Information);

            var uniqueSet = _messages.GetOrAdd(messageType, new UniqueTypeSet());
            uniqueSet[typeof(T)] = message;
        }

        public T RetrieveMessage<T>(MessageType messageType)
        {
            if (_messages.TryGetValue(messageType, out var uniqueSet) && uniqueSet.ContainsType(typeof(T)))
            {
                return (T)uniqueSet[typeof(T)];
            }

            throw new InvalidOperationException($"Message of type [{messageType}] with data type [{typeof(T).Name}] not found.");
        }

        public bool RemoveMessage(MessageType messageType)
        {
            return _messages.TryRemove(messageType, out _);
        }
    }
}
