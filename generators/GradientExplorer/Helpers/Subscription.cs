using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GradientExplorer.Helpers
{
    public class Subscription<T> : SubscriptionBase, IDisposable where T : IEventData
    {
        private readonly Action<T, CancellationToken> _action;
        private readonly Func<T, bool> _filter;

        private readonly ConcurrentDictionary<int, ThreadSafeList<SubscriptionBase>> _subscribers;

        public Subscription(Action<T, CancellationToken> action, int priority, Func<T, bool> filter, ConcurrentDictionary<int, ThreadSafeList<SubscriptionBase>> subscribers) : base(priority)
        {
            _action = action;
            _filter = filter;
            _subscribers = subscribers;

            if (!_subscribers.ContainsKey(Priority))
            {
                _subscribers[Priority] = new ThreadSafeList<SubscriptionBase>();
            }

            _subscribers[Priority].Add(this);
        }

        public Action<T, CancellationToken> Action => _action;

        public Func<T, bool> Filter => _filter;

        public ConcurrentDictionary<int, ThreadSafeList<SubscriptionBase>> Subscribers => _subscribers;

        public void Dispose()
        {
            base.Dispose();
            _subscribers[Priority].Remove(this);
        }
    }
}
