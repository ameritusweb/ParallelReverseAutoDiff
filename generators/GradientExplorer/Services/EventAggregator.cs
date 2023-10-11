using GradientExplorer.Helpers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace GradientExplorer.Services
{
    public class EventAggregator : IEventAggregator
    {
        private readonly ConcurrentDictionary<EventType, ConcurrentDictionary<int, List<SubscriptionBase>>> _syncSubscriptions = new ConcurrentDictionary<EventType, ConcurrentDictionary<int, List<SubscriptionBase>>>();
        private readonly ConcurrentDictionary<EventType, ConcurrentDictionary<int, List<SubscriptionBase>>> _asyncSubscriptions = new ConcurrentDictionary<EventType, ConcurrentDictionary<int, List<SubscriptionBase>>>();
        private readonly ConcurrentDictionary<MessageType, UniqueTypeSet> _messages = new ConcurrentDictionary<MessageType, UniqueTypeSet>();

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

        public void Publish<T>(EventType eventType, T eventData) where T : IEventData
        {
            // Handle synchronous subscribers
            if (_syncSubscriptions.TryGetValue(eventType, out var syncSubscribers))
            {
                InvokeSubscribers(eventType, syncSubscribers, eventData);
            }

            // Handle asynchronous subscribers
            if (_asyncSubscriptions.TryGetValue(eventType, out var asyncSubscribers))
            {
                InvokeAsyncSubscribers(eventType, asyncSubscribers, eventData).Wait();
            }
        }

        public void PostMessage<T>(MessageType messageType, T message)
        {
            var uniqueSet = _messages.GetOrAdd(messageType, new UniqueTypeSet());
            uniqueSet[typeof(T)] = message;
        }

        public T RetrieveMessage<T>(MessageType messageType)
        {
            if (_messages.TryGetValue(messageType, out var uniqueSet) && uniqueSet.ContainsType(typeof(T)))
            {
                return (T)uniqueSet[typeof(T)];
            }

            throw new InvalidOperationException($"Message of type [{messageType}] not found.");
        }

        public bool RemoveMessage(MessageType messageType)
        {
            return _messages.TryRemove(messageType, out _);
        }

        private void InvokeSubscribers<T>(EventType eventType, ConcurrentDictionary<int, List<SubscriptionBase>> subscribers, T eventData) where T : IEventData
        {
            foreach (var priorityGroup in subscribers.OrderByDescending(kvp => kvp.Key))
            {
                foreach (var subscription in priorityGroup.Value.OfType<Subscription<T>>())
                {
                    if (subscription.CancellationTokenSource.Token.IsCancellationRequested)
                    {
                        continue; // Skip if cancellation has been requested
                    }

                    if (subscription.Filter == null || subscription.Filter(eventData))
                    {
                        subscription.Stopwatch.Start();

                        try
                        {
                            subscription.Action(eventData, subscription.CancellationTokenSource.Token);
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Encountered an exception while invoking a subscriber for event type {eventType}: {ex.Message}");
                        }
                        finally
                        {
                            subscription.Stopwatch.Stop();
                        }
                    }
                }
            }
        }

        private async Task InvokeAsyncSubscribers<T>(EventType eventType, ConcurrentDictionary<int, List<SubscriptionBase>> asyncSubscribers, T eventData) where T : IEventData
        {
            foreach (var priorityGroup in asyncSubscribers.OrderByDescending(kvp => kvp.Key))
            {
                var tasks = new List<Task>();

                foreach (var subscription in priorityGroup.Value.OfType<SubscriptionAsync<T>>())
                {
                    if (subscription.CancellationTokenSource.Token.IsCancellationRequested)
                    {
                        continue; // Skip if cancellation has been requested
                    }

                    if (subscription.Filter == null || subscription.Filter(eventData))
                    {
                        subscription.Stopwatch.Start();

                        tasks.Add(Task.Run(async () =>
                        {
                            try
                            {
                                await subscription.AsyncAction(eventData, subscription.CancellationTokenSource.Token).ConfigureAwait(false);
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Encountered an exception while invoking a subscriber for event type {eventType}: {ex.Message}");
                            }
                            finally
                            {
                                subscription.Stopwatch.Stop();
                            }
                        }));
                    }
                }

                await Task.WhenAll(tasks).ConfigureAwait(false);
            }
        }
    }
}
