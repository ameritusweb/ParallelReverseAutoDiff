using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media;

namespace GradientExplorer.Helpers
{
    public class BackgroundEventData : IEventData
    {
        private CancellationTokenSource _cancellationTokenSource = new CancellationTokenSource();
        public CancellationTokenSource CancellationTokenSource => _cancellationTokenSource;

        public SolidColorBrush SolidColorBrush { get; set; }
    }
}
