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
    public class SubscriptionAsync<T> : SubscriptionBase, IDisposable where T : IEventData
    {
        private readonly Func<T, CancellationToken, Task> _asyncAction;
        private readonly Func<T, bool> _filter;

        private readonly ConcurrentDictionary<int, List<SubscriptionBase>> _asyncSubscribers;

        public SubscriptionAsync(Func<T, CancellationToken, Task> asyncAction, int priority, Func<T, bool> filter, ConcurrentDictionary<int, List<SubscriptionBase>> asyncSubscribers): base(priority)
        {
            _asyncAction = asyncAction;
            _filter = filter;
            _asyncSubscribers = asyncSubscribers;

            if (!_asyncSubscribers.ContainsKey(Priority))
            {
                _asyncSubscribers[Priority] = new List<SubscriptionBase>();
            }

            _asyncSubscribers[Priority].Add(this);
        }

        public Func<T, CancellationToken, Task> AsyncAction => _asyncAction;

        public Func<T, bool> Filter => _filter;

        public void Dispose()
        {
            base.Dispose();
            _asyncSubscribers[Priority].Remove(this);
        }
    }
}
