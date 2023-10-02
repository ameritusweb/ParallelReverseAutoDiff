using CSharpMath.Rendering.FrontEnd;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;

namespace GradientExplorer.LaTeX.Translation
{
    public class WpfPath : Path
    {
        private StreamGeometry geometry = new StreamGeometry();
        private StreamGeometryContext context;
        private ICanvas _canvas;

        public WpfPath(ICanvas canvas)
        {
            _canvas = canvas;
        }

        public void BeginRead(int contourCount)
        {
            context = geometry.Open();
        }

        public void EndRead()
        {
            context.Close();
        }

        public override void MoveTo(float x0, float y0)
        {
            context.BeginFigure(new Point(x0, _canvas.Height - y0), true /* is filled */, false /* is closed */);
        }

        public override void LineTo(float x1, float y1)
        {
            context.LineTo(new Point(x1, _canvas.Height - y1), true /* is stroked */, false /* is smooth join */);
        }

        public override void Curve3(float x1, float y1, float x2, float y2)
        {
            // For Quadratic Bezier, we need to calculate the control point
            // Here it's simply passed as is
            context.QuadraticBezierTo(new Point(x1, _canvas.Height - y1), new Point(x2, _canvas.Height - y2), true /* is stroked */, false /* is smooth join */);
        }

        public override void Curve4(float x1, float y1, float x2, float y2, float x3, float y3)
        {
            // For Cubic Bezier
            context.BezierTo(new Point(x1, _canvas.Height - y1), new Point(x2, _canvas.Height - y2), new Point(x3, _canvas.Height - y3), true /* is stroked */, false /* is smooth join */);
        }

        public override void CloseContour()
        {
            // In WPF, contours are automatically closed when you fill a geometry,
            // so you may not need to do anything special here.
        }

        public override void Dispose()
        {
            // Dispose of WPF resources if needed
        }

        public override System.Drawing.Color? Foreground { get; set; }

        public StreamGeometry GetGeometry()
        {
            return geometry;
        }
    }
}
