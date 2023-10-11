using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Shapes;

namespace GradientExplorer.Helpers
{
    public class PathEventData : IEventData
    {
        private CancellationTokenSource _cancellationTokenSource = new CancellationTokenSource();
        public CancellationTokenSource CancellationTokenSource => _cancellationTokenSource;

        public Path Path { get; set; }

        public float Top { get; set; }

        public float Left { get; set; }
    }
}
