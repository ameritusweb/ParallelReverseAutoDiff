using GradientExplorer.Helpers;
using System.Collections.Concurrent;
using System.Threading;

namespace GradientExplorer.Services
{
    public class EventAggregator : IEventAggregator, IMessagePoster, IMessageRetriever
    {
        private readonly ConcurrentDictionary<EventType, ConcurrentDictionary<int, ThreadSafeList<SubscriptionBase>>> _syncSubscriptions = new ConcurrentDictionary<EventType, ConcurrentDictionary<int, ThreadSafeList<SubscriptionBase>>>();
        private readonly ConcurrentDictionary<EventType, ConcurrentDictionary<int, ThreadSafeList<SubscriptionBase>>> _asyncSubscriptions = new ConcurrentDictionary<EventType, ConcurrentDictionary<int, ThreadSafeList<SubscriptionBase>>>();
        private readonly ConcurrentDictionary<MessageType, UniqueTypeSet> _messages = new ConcurrentDictionary<MessageType, UniqueTypeSet>();
        private readonly ConcurrentDictionary<Type, object> _invokerCache = new ConcurrentDictionary<Type, object>();
        private readonly ILogger _logger;

        public EventAggregator(ILogger logger)
        {
            _logger = logger;
        }

        public Subscription<T> Subscribe<T>(EventType eventType, Action<T, CancellationToken> action, int priority, Func<T, bool> filter = null) where T : IEventData
        {
            var subscribers = _syncSubscriptions.GetOrAdd(eventType, _ => new ConcurrentDictionary<int, ThreadSafeList<SubscriptionBase>>());

            var subscription = new Subscription<T>(action, priority, filter, subscribers);
            if (!subscribers.ContainsKey(priority))
            {
                subscribers[priority] = new ThreadSafeList<SubscriptionBase>();
            }

            subscribers[priority].Add(subscription);

            return subscription;
        }

        public SubscriptionAsync<T> SubscribeAsync<T>(EventType eventType, Func<T, CancellationToken, Task> asyncAction, int priority, Func<T, bool> filter = null) where T : IEventData
        {
            var asyncSubscribers = _asyncSubscriptions.GetOrAdd(eventType, _ => new ConcurrentDictionary<int, ThreadSafeList<SubscriptionBase>>());

            var subscription = new SubscriptionAsync<T>(asyncAction, priority, filter, asyncSubscribers);
            if (!asyncSubscribers.ContainsKey(priority))
            {
                asyncSubscribers[priority] = new ThreadSafeList<SubscriptionBase>();
            }

            asyncSubscribers[priority].Add(subscription);

            return subscription;
        }

        public async Task PublishAsync<T>(EventType eventType, T eventData) where T : IEventData
        {
            _logger.Log($"Message published with event type [{eventType}]", SeverityType.Information);

            bool foundSubscribers = false;

            // Retrieve or create the invoker from the cache.
            ISubscriptionInvoker<T> invoker = (ISubscriptionInvoker<T>)_invokerCache.GetOrAdd(
                typeof(T),
                _ => SubscriptionInvokerFactory.GetInvoker<T>(_logger)
            );

            // Handle synchronous subscribers
            if (_syncSubscriptions.TryGetValue(eventType, out var syncSubscribers))
            {
                foundSubscribers = true;
                invoker.InvokeSubscribers(eventType, syncSubscribers, eventData);

            }

            // Handle asynchronous subscribers
            if (_asyncSubscriptions.TryGetValue(eventType, out var asyncSubscribers))
            {
                foundSubscribers = true;
                await invoker.InvokeSubscribersAsync(eventType, asyncSubscribers, eventData);
            }

            if (!foundSubscribers && !eventData.Options.AllowNoSubscribers)
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
