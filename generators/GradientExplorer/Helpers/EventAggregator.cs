using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace GradientExplorer.Helpers
{
    public class EventAggregator : IEventAggregator
    {
        private class Subscription : IComparable<Subscription>
        {
            public Delegate Handler { get; set; }
            public int Priority { get; set; }

            public int CompareTo(Subscription other)
            {
                return other.Priority.CompareTo(this.Priority);
            }
        }

        private readonly ConcurrentDictionary<Type, SortedSet<Subscription>> _subscribers =
            new ConcurrentDictionary<Type, SortedSet<Subscription>>();

        // New member for memoization
        private readonly ConcurrentDictionary<Type, List<MethodInfo>> _methodInfoCache =
            new ConcurrentDictionary<Type, List<MethodInfo>>();

        public void Publish<TEvent>(TEvent eventToPublish)
        {
            if (_subscribers.TryGetValue(typeof(TEvent), out var subscriptions))
            {
                foreach (var subscription in subscriptions)  // Already sorted due to SortedSet
                {
                    if (subscription.Handler is Action<TEvent> handler)
                    {
                        handler(eventToPublish);
                    }
                    else
                    {
                        try
                        {
                            subscription.Handler.DynamicInvoke(eventToPublish);
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error occurred while handling event {typeof(TEvent).Name}: {ex.Message}");
                        }
                    }
                }
            }
        }

        public void Subscribe(object subscriber, int priority = 0)
        {
            var subscriberType = subscriber.GetType();
            var methods = _methodInfoCache.GetOrAdd(subscriberType, st =>
                st.GetMethods().Where(m => m.GetCustomAttributes(typeof(HandlesAttribute), false).Length > 0).ToList()
            );
            foreach (var method in subscriberType.GetMethods())
            {
                var parameters = method.GetParameters();
                if (parameters.Length == 1 && method.GetCustomAttributes(typeof(HandlesAttribute), false).Length > 0)
                {
                    var eventType = parameters[0].ParameterType;
                    var newSubscription = new Subscription
                    {
                        Handler = Delegate.CreateDelegate(typeof(Action<>).MakeGenericType(eventType), subscriber, method),
                        Priority = priority
                    };

                    _subscribers.AddOrUpdate(eventType,
                        _ => new SortedSet<Subscription> { newSubscription },
                        (_, existing) =>
                        {
                            existing.Add(newSubscription);
                            return existing;
                        });
                }
            }
        }

        public void Unsubscribe(object subscriber, int? priority = null, Type eventType = null)
        {
            var subscriberType = subscriber.GetType();
            var methods = _methodInfoCache.GetOrAdd(subscriberType, st =>
                st.GetMethods().Where(m => m.GetCustomAttributes(typeof(HandlesAttribute), false).Length > 0).ToList()
            );

            foreach (var method in methods)
            {
                var handlesAttributes = method.GetCustomAttributes(typeof(HandlesAttribute), false);
                foreach (HandlesAttribute attr in handlesAttributes)
                {
                    var currentEventType = attr.EventType;
                    if (eventType != null && currentEventType != eventType)
                    {
                        continue;
                    }

                    if (_subscribers.TryGetValue(currentEventType, out var existingSubscriptions))
                    {
                        var delegateToRemove = Delegate.CreateDelegate(typeof(Action<>).MakeGenericType(currentEventType), subscriber, method);
                        var subscriptionsToRemove = existingSubscriptions.Where(s => s.Handler == delegateToRemove && (priority == null || s.Priority == priority)).ToList();

                        foreach (var subscription in subscriptionsToRemove)
                        {
                            existingSubscriptions.Remove(subscription);
                        }
                    }
                }
            }
        }
    }
}
