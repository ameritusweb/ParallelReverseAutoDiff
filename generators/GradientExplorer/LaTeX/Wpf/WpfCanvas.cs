using CSharpMath.Rendering.FrontEnd;
using GradientExplorer.LaTeX.Translation;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace GradientExplorer.LaTeX.Wpf
{
    public class WpfCanvas : ICanvas
    {
        private SolidColorBrush _currentColorBrush;
        private Canvas _canvas;
        private double _currentX;
        private double _currentY;

        public WpfCanvas(Canvas canvas)
        {
            _canvas = canvas;
            _currentY = canvas.ActualHeight;
        }   

        public System.Drawing.Color DefaultColor { get; set; }

        public System.Drawing.Color? CurrentColor
        {
            get => _currentColorBrush != null ? System.Drawing.Color.FromArgb(_currentColorBrush.Color.A, _currentColorBrush.Color.R, _currentColorBrush.Color.G, _currentColorBrush.Color.B) : (System.Drawing.Color?)null;
            set
            {
                if (value.HasValue)
                {
                    _currentColorBrush = new SolidColorBrush(System.Windows.Media.Color.FromArgb(value.Value.A, value.Value.R, value.Value.G, value.Value.B));
                    _currentColorBrush.Freeze();
                }
                else
                {
                    _currentColorBrush = null;
                }
            }
        }

        public PaintStyle CurrentStyle { get; set; }

        float ICanvas.Width => (float)_canvas.ActualWidth;

        float ICanvas.Height => (float)_canvas.ActualHeight;

        public void DrawLine(float x1, float y1, float x2, float y2, float lineThickness)
        {
            var line = new Line
            {
                X1 = x1,
                Y1 = y1,
                X2 = x2,
                Y2 = y2,
                Stroke = _currentColorBrush,
                StrokeThickness = lineThickness
            };
            _canvas.Children.Add(line);
        }

        public void FillRect(float left, float top, float width, float height)
        {
            var rect = new System.Windows.Shapes.Rectangle
            {
                Width = width,
                Height = height,
                Fill = _currentColorBrush
            };
            Canvas.SetLeft(rect, left);
            Canvas.SetTop(rect, top);
            _canvas.Children.Add(rect);
        }

        // Add this new method to render StreamGeometry
        public void DrawStreamGeometry(StreamGeometry geometry, System.Windows.Media.Color color)
        {
            // Create a Path to hold the geometry and set properties
            var path = new System.Windows.Shapes.Path();
            path.Data = geometry;
            path.Stroke = new SolidColorBrush(color);
            path.StrokeThickness = 1; // You can adjust this value as needed

            Canvas.SetTop(path, _currentY);
            Canvas.SetLeft(path, _currentX);

            // Add the Path to the Canvas' children
            _canvas.Children.Add(path);
        }

        public void Restore()
        {
            // Implement state restoration if needed
        }

        public void Save()
        {
            // Implement state saving if needed
        }

        public void Scale(float sx, float sy)
        {
            // Implement scaling if needed
        }

        public CSharpMath.Rendering.FrontEnd.Path StartNewPath()
        {
            // Use your WpfPath implementation here
            return new WpfPath();
        }

        public void StrokeRect(float left, float top, float width, float height)
        {
            var rect = new System.Windows.Shapes.Rectangle
            {
                Width = width,
                Height = height,
                Stroke = _currentColorBrush
            };
            Canvas.SetLeft(rect, left);
            Canvas.SetTop(rect, top);
            _canvas.Children.Add(rect);
        }

        public void Translate(float dx, float dy)
        {
            _currentX += dx;
            _currentY += dy;
        }

        public void SetTextPosition(float fx, float dy)
        {
            // Implement text positioning if needed
        }
    }
}
