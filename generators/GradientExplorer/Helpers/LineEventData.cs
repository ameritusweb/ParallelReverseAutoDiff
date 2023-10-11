using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Shapes;

namespace GradientExplorer.Helpers
{
    public class LineEventData : IEventData
    {
        private CancellationTokenSource _cancellationTokenSource = new CancellationTokenSource();
        public CancellationTokenSource CancellationTokenSource => _cancellationTokenSource;

        public Line Line { get; set; }
    }
}
