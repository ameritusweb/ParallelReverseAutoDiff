using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Shapes;

namespace GradientExplorer.Helpers
{
    public class PanelLayoutTransformEventData : IEventData
    {
        private CancellationTokenSource _cancellationTokenSource = new CancellationTokenSource();
        public CancellationTokenSource CancellationTokenSource => _cancellationTokenSource;

        public Transform LayoutTransform { get; set; }
    }
}
