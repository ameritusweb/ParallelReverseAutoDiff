using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Shapes;

namespace GradientExplorer.Helpers
{
    public class AddPathCanvasEvent : CanvasEventBase
    {
        // Properties or methods specific to adding a path
        public Path PathToAdd { get; set; }
    }
}
