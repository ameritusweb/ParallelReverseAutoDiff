using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GradientExplorer.Helpers
{
    public interface IEventData
    {
        public CancellationTokenSource CancellationTokenSource { get; }

        public PublishOptions Options { get; }
    }
}
