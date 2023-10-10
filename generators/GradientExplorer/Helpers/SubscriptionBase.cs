using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GradientExplorer.Helpers
{
    public abstract class SubscriptionBase
    {
        public int Priority { get; protected set; }
        public Stopwatch Stopwatch { get; protected set; }
        public CancellationTokenSource CancellationTokenSource { get; protected set; }

        public SubscriptionBase(int priority)
        {
            Priority = priority;
            Stopwatch = new Stopwatch();
            CancellationTokenSource = new CancellationTokenSource();
        }

        public virtual void Dispose()
        {
            CancellationTokenSource.Dispose();
        }
    }
}
