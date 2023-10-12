using GradientExplorer.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Helpers
{
    public static class SubscriptionInvokerFactory
    {
        public static ISubscriptionInvoker<T> GetInvoker<T>(ILogger logger) where T : IEventData
        {
            return new SubscriptionInvoker<T>(logger);  // Assume you've implemented this class.
        }
    }
}
