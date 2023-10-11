using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GradientExplorer.Helpers
{
    public class ClearEventData : IEventData
    {
        private CancellationTokenSource _cancellationTokenSource = new CancellationTokenSource();
        public CancellationTokenSource CancellationTokenSource => _cancellationTokenSource;
    }
}
