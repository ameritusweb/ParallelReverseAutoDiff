using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Helpers
{
    public class SubscriptionQueue<T> : ConcurrentQueue<T> where T : ISubscriptionBase
    {

    }
}
