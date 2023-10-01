using CSharpMath.Rendering.FrontEnd;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Media;

namespace GradientExplorer.LaTeX.Wpf
{
    public class WpfMathPainter : MathPainter<WpfCanvas, SolidColorBrush>
    {
        public override SolidColorBrush UnwrapColor(System.Drawing.Color color)
        {
            throw new NotImplementedException();
        }

        public override ICanvas WrapCanvas(WpfCanvas canvas)
        {
            throw new NotImplementedException();
        }

        public override System.Drawing.Color WrapColor(SolidColorBrush color)
        {
            throw new NotImplementedException();
        }
    }
}
