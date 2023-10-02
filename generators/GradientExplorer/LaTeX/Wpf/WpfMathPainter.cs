using CSharpMath.Rendering.FrontEnd;
using System.Windows.Media;

namespace GradientExplorer.LaTeX.Wpf
{
    public class WpfMathPainter : MathPainter<WpfCanvas, SolidColorBrush>
    {
        public override SolidColorBrush UnwrapColor(System.Drawing.Color color)
        {
            if (color.IsEmpty)
            {
                return null; // Or a default SolidColorBrush, as you prefer
            }

            return new SolidColorBrush(
                Color.FromArgb(
                    color.A,
                    color.R,
                    color.G,
                    color.B
                )
            );
        }

        public override ICanvas WrapCanvas(WpfCanvas canvas)
        {
            return canvas;
        }

        public override System.Drawing.Color WrapColor(SolidColorBrush color)
        {
            if (color == null)
            {
                return System.Drawing.Color.Empty;
            }

            return System.Drawing.Color.FromArgb(
                color.Color.A,
                color.Color.R,
                color.Color.G,
                color.Color.B
            );
        }
    }
}
