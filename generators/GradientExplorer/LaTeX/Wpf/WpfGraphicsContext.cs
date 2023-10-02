using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Media;
using CSharpMath.Display.FrontEnd;
using CSharpMath.Rendering.FrontEnd;
using CSharpMath.Structures;
using GradientExplorer.LaTeX.Translation;
using GradientExplorer.LaTeX.Wpf;
using Typography.OpenFont;

namespace CSharpMath.Rendering.BackEnd
{
    public class WpfGraphicsContext : IGraphicsContext<Fonts, Glyph>
    {
        private class GlyphOutlineBuilder : Typography.Contours.GlyphOutlineBuilderBase
        {
            public GlyphOutlineBuilder(Typography.OpenFont.Typeface typeface) : base(typeface) { }
        }

        public WpfGraphicsContext(ICanvas canvas, (System.Drawing.Color glyph, System.Drawing.Color textRun)? glyphBoxColor)
        {
            Canvas = canvas;
            GlyphBoxColor = glyphBoxColor;
        }
        public (System.Drawing.Color glyph, System.Drawing.Color textRun)? GlyphBoxColor { get; set; }
        public ICanvas Canvas { get; set; }
        void IGraphicsContext<Fonts, Glyph>.SetTextPosition(PointF position) => Canvas.SetTextPosition(position.X, position.Y);
        public void DrawGlyphsAtPoints
          (IReadOnlyList<Glyph> glyphs, Fonts font, IEnumerable<PointF> points, System.Drawing.Color? color)
        {
            if (color == null)
            {
                color = System.Drawing.Color.White;
            }
            foreach (var (glyph, point) in glyphs.Zip(points, System.ValueTuple.Create))
            {
                if (GlyphBoxColor != null)
                {
                    using var rentedArray = new RentedArray<Glyph>(glyph);
                    var rect =
                      GlyphBoundsProvider.Instance.GetBoundingRectsForGlyphs(font, rentedArray.Result, 1).Single();
                    Canvas.CurrentColor = GlyphBoxColor?.glyph;
                    Canvas.StrokeRect(point.X + rect.X, point.Y + rect.Y, rect.Width, rect.Height);
                }
                var typeface = glyph.Typeface;
                var scale = typeface.CalculateScaleToPixelFromPointSize(font.PointSize);
                // var pathBuilder = new GlyphOutlineBuilder(typeface);
                // pathBuilder.BuildFromGlyph(glyph.Info, font.PointSize);
                Canvas.Save();
                Canvas.CurrentColor = color;
                Canvas.Translate(point.X, point.Y);
                var wpfPath = new WpfPath();
                wpfPath.BeginRead(0);  // The contour count is not being used in your implementation
                var pathBuilder = new GlyphOutlineBuilder(glyph.Typeface);
                pathBuilder.BuildFromGlyph(glyph.Info, font.PointSize);
                pathBuilder.ReadShapes(wpfPath);
                wpfPath.EndRead();

                StreamGeometry geometry = wpfPath.GetGeometry();
                // Now use 'geometry' to draw the path onto your WPF canvas
                ((WpfCanvas)Canvas).DrawStreamGeometry(geometry, System.Windows.Media.Color.FromArgb(Canvas.CurrentColor.Value.A, Canvas.CurrentColor.Value.R, Canvas.CurrentColor.Value.G, Canvas.CurrentColor.Value.B));
                Canvas.Restore();
            }
        }
        public void DrawLine(float x1, float y1, float x2, float y2, float lineThickness, System.Drawing.Color? color)
        {
            Canvas.CurrentColor = color;
            Canvas.DrawLine(x1, y1, x2, y2, lineThickness);
        }
        public void DrawGlyphRunWithOffset
          (Display.AttributedGlyphRun<Fonts, Glyph> run, PointF offset, System.Drawing.Color? color)
        {
            if (color == null)
            {
                color = System.Drawing.Color.White;
            }
            var textPosition = offset;
            if (GlyphBoxColor != null)
            {
                Bounds bounds;
                float scale, ascent = 0, descent = 0;
                foreach (var (glyph, kernAfter, _) in run.GlyphInfos)
                {
                    bounds = glyph.Info.Bounds;
                    scale = glyph.Typeface.CalculateScaleToPixelFromPointSize(run.Font.PointSize);
                    ascent = System.Math.Max(ascent, bounds.YMax * scale);
                    descent = System.Math.Min(descent, bounds.YMin * scale);
                }
                var width = GlyphBoundsProvider.Instance.GetTypographicWidth(run.Font, run);
                Canvas.CurrentColor = GlyphBoxColor?.textRun;
                Canvas.StrokeRect(textPosition.X, textPosition.Y + descent, width, ascent - descent);
            }
            var pointSize = run.Font.PointSize;
            Canvas.Save();
            Canvas.Translate(textPosition.X, textPosition.Y);
            Canvas.CurrentColor = color;
            foreach (var (glyph, kernAfter, foreground) in run.GlyphInfos)
            {
                var typeface = glyph.Typeface;
                var scale = typeface.CalculateScaleToPixelFromPointSize(pointSize);
                var index = glyph.Info.GlyphIndex;
                Canvas.CurrentColor = foreground ?? color;
                var wpfPath = new WpfPath();
                wpfPath.BeginRead(0);  // The contour count is not being used in your implementation
                var pathBuilder = new GlyphOutlineBuilder(glyph.Typeface);
                pathBuilder.BuildFromGlyph(glyph.Info, pointSize);
                pathBuilder.ReadShapes(wpfPath);
                wpfPath.EndRead();

                StreamGeometry geometry = wpfPath.GetGeometry();
                // Now use 'geometry' to draw the path onto your WPF canvas
                ((WpfCanvas)Canvas).DrawStreamGeometry(geometry, System.Windows.Media.Color.FromArgb(Canvas.CurrentColor.Value.A, Canvas.CurrentColor.Value.R, Canvas.CurrentColor.Value.G, Canvas.CurrentColor.Value.B));
                Canvas.Translate(typeface.GetHAdvanceWidthFromGlyphIndex(index) * scale + kernAfter, 0);
            }
            Canvas.Restore();
        }

        public void FillRect(RectangleF rect, System.Drawing.Color color)
        {
            Canvas.CurrentColor = color;
            Canvas.FillRect(rect.X, rect.Y, rect.Width, rect.Height);
        }
        public void RestoreState() => Canvas.Restore();
        public void SaveState() => Canvas.Save();
        public void Translate(PointF dxy) => Canvas.Translate(dxy.X, dxy.Y);
    }
}
