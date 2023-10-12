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
    public class PanelLayoutTransformEventData : EventDataBase
    {

        public Transform LayoutTransform { get; set; }
    }
}
