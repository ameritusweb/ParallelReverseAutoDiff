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
        private readonly Action<T> _action;
        private readonly Func<T, bool> _filter;

        private readonly ConcurrentDictionary<int, List<SubscriptionBase>> _subscribers;

        public Subscription(Action<T> action, int priority, Func<T, bool> filter, ConcurrentDictionary<int, List<SubscriptionBase>> subscribers) : base(priority)
        {
            _action = action;
            _filter = filter;
            _subscribers = subscribers;

            if (!_subscribers.ContainsKey(Priority))
            {
                _subscribers[Priority] = new List<SubscriptionBase>();
            }

            _subscribers[Priority].Add(this);
        }

        public Action<T> Action => _action;

        public Func<T, bool> Filter => _filter;

        public ConcurrentDictionary<int, List<SubscriptionBase>> Subscribers => _subscribers;

        public void Dispose()
        {
            base.Dispose();
            _subscribers[Priority].Remove(this);
        }
    }
}
