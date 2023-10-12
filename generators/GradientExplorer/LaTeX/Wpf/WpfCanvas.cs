using CSharpMath.Rendering.FrontEnd;
using GradientExplorer.Helpers;
using GradientExplorer.LaTeX.Translation;
using GradientExplorer.Services;
using Microsoft.VisualStudio.PlatformUI;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Shapes;

namespace GradientExplorer.LaTeX.Wpf
{
    public class WpfCanvas : ICanvas
    {
        private SolidColorBrush _defaultColorBrush;
        private SolidColorBrush _currentColorBrush;
        private double _currentX;
        private double _currentY;
        private Dictionary<Guid, (double, double)> _savedMap;
        private readonly IEventAggregator eventAggregator;
        private readonly IMessageRetriever messageRetriever;

        public WpfCanvas(IEventAggregator eventAggregator, IMessageRetriever messageRetriever)
        {
            _currentY = 0;
            _savedMap = new Dictionary<Guid, (double, double)>();
            this.eventAggregator = eventAggregator;
            var backgroundColor = VSColorTheme.GetThemedColor(EnvironmentColors.ToolWindowBackgroundColorKey);
            eventAggregator.PublishAsync(EventType.SetCanvasBackground, new BackgroundEventData { SolidColorBrush = new SolidColorBrush(System.Windows.Media.Color.FromArgb(backgroundColor.A, backgroundColor.R, backgroundColor.G, backgroundColor.B)) }).Wait();
            this.messageRetriever = messageRetriever;
        }

        public void SetWidth(float width)
        {
            eventAggregator.PublishAsync(EventType.SetCanvasWidth, new WidthEventData { Width = width }).Wait();
        }

        public System.Drawing.Color DefaultColor
        {
            get => System.Drawing.Color.FromArgb(_defaultColorBrush.Color.A, _defaultColorBrush.Color.R, _defaultColorBrush.Color.G, _defaultColorBrush.Color.B);
            set => _defaultColorBrush = new SolidColorBrush(System.Windows.Media.Color.FromArgb(value.A, value.R, value.G, value.B));
        }

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

        float ICanvas.Width
        {
            get
            {
                return messageRetriever.RetrieveMessage<float>(MessageType.CanvasWidth);
            }
        }

        float ICanvas.Height
        {
            get
            {
                return messageRetriever.RetrieveMessage<float>(MessageType.CanvasHeight);
            }
        }

        public void DrawLine(float x1, float y1, float x2, float y2, float lineThickness)
        {
            var line = new Line
            {
                X1 = _currentX + x1,
                Y1 = _currentY - y1,
                X2 = _currentX + x2,
                Y2 = _currentY - y2,
                Stroke = _currentColorBrush ?? new SolidColorBrush(Colors.White),
                StrokeThickness = lineThickness
            };
            eventAggregator.PublishAsync(EventType.AddLineToCanvas, new LineEventData { Line = line }).Wait();
        }

        public void FillRect(float left, float top, float width, float height)
        {
            var rect = new System.Windows.Shapes.Rectangle
            {
                Width = width,
                Height = height,
                Fill = _currentColorBrush
            };
            float actualHeight = messageRetriever.RetrieveMessage<float>(MessageType.CanvasActualHeight);
            eventAggregator.PublishAsync(EventType.AddRectToCanvas, new RectEventData { Rect = rect, Top = actualHeight + top, Left = left }).Wait();
        }

        // Add this new method to render StreamGeometry
        public void DrawStreamGeometry(StreamGeometry geometry, System.Windows.Media.Color color)
        {
            // Create a Path to hold the geometry and set properties
            var path = new System.Windows.Shapes.Path();
            path.Data = geometry;
            path.Stroke = new SolidColorBrush(color);
            path.StrokeThickness = 0.5; // You can adjust this value as needed

            var actualHeight = messageRetriever.RetrieveMessage<float>(MessageType.CanvasActualHeight);
            eventAggregator.PublishAsync(EventType.AddPathToCanvas, new PathEventData { Path = path, Top = (float)(_currentY - actualHeight), Left = (float)_currentX }).Wait();
        }

        public void Restore(Guid id)
        {
            // Implement state restoration if needed
            var restored = _savedMap[id];
            this._currentX = restored.Item1;
            this._currentY = restored.Item2;
        }

        public void Save(Guid id)
        {
            // Implement state saving if needed
            _savedMap[id] = (this._currentX, this._currentY);
        }

        public void Scale(float sx, float sy)
        {
            // Implement scaling if needed
        }

        public CSharpMath.Rendering.FrontEnd.Path StartNewPath()
        {
            // Use your WpfPath implementation here
            return new WpfPath(this);
        }

        public void StrokeRect(float left, float top, float width, float height)
        {
            var rect = new System.Windows.Shapes.Rectangle
            {
                Width = width,
                Height = height,
                Stroke = _currentColorBrush
            };
            eventAggregator.PublishAsync(EventType.AddRectToCanvas, new RectEventData { Rect = rect, Top = top, Left = left }).Wait();
        }

        public void Translate(float dx, float dy)
        {
            _currentX += dx;
            _currentY -= dy;
        }

        public void SetTextPosition(float fx, float dy)
        {
            // Implement text positioning if needed
            _currentX += fx;
            _currentY -= dy;
        }
    }
}
