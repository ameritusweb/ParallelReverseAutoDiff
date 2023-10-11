using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GradientExplorer.Helpers
{
    public interface ISubscriptionBase : IDisposable
    {
        int Priority { get; }
        Stopwatch Stopwatch { get; }
        CancellationTokenSource CancellationTokenSource { get; }
        new void Dispose();
    }
}
