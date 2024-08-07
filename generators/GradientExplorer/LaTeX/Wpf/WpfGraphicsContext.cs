﻿using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Media;
using CSharpMath.Display.FrontEnd;
using CSharpMath.Rendering.FrontEnd;
using CSharpMath.Structures;
using GradientExplorer.LaTeX.Translation;
using GradientExplorer.LaTeX.Wpf;
using Microsoft.VisualStudio.PlatformUI;
using Typography.OpenFont;

namespace CSharpMath.Rendering.BackEnd
{
    public class WpfGraphicsContext : IGraphicsContext<Fonts, Glyph>
    {
        private System.Drawing.Color defaultColor;

        private class GlyphOutlineBuilder : Typography.Contours.GlyphOutlineBuilderBase
        {
            public GlyphOutlineBuilder(Typography.OpenFont.Typeface typeface) : base(typeface) { }
        }

        public WpfGraphicsContext(ICanvas canvas, (System.Drawing.Color glyph, System.Drawing.Color textRun)? glyphBoxColor)
        {
            Canvas = canvas;
            GlyphBoxColor = glyphBoxColor;
            var foregroundColor = VSColorTheme.GetThemedColor(EnvironmentColors.ToolWindowTextColorKey);
            defaultColor = System.Drawing.Color.FromArgb(foregroundColor.A, foregroundColor.R, foregroundColor.G, foregroundColor.B);
        }
        public (System.Drawing.Color glyph, System.Drawing.Color textRun)? GlyphBoxColor { get; set; }
        public ICanvas Canvas { get; set; }
        void IGraphicsContext<Fonts, Glyph>.SetTextPosition(PointF position) => Canvas.SetTextPosition(position.X, position.Y);
        public void DrawGlyphsAtPoints
          (IReadOnlyList<Glyph> glyphs, Fonts font, IEnumerable<PointF> points, System.Drawing.Color? color)
        {
            if (color == null)
            {
                color = defaultColor;
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
                typeface.CalculateScaleToPixelFromPointSize(font.PointSize);
                Guid id = Guid.NewGuid();
                Canvas.Save(id);
                Canvas.CurrentColor = color;
                Canvas.Translate(point.X, point.Y);
                var wpfPath = new WpfPath(Canvas);
                wpfPath.BeginRead(0);  // The contour count is not being used in your implementation
                var pathBuilder = new GlyphOutlineBuilder(glyph.Typeface);
                pathBuilder.BuildFromGlyph(glyph.Info, font.PointSize);
                pathBuilder.ReadShapes(wpfPath);
                wpfPath.EndRead();

                StreamGeometry geometry = wpfPath.GetGeometry();
                // Now use 'geometry' to draw the path onto your WPF canvas
                ((WpfCanvas)Canvas).DrawStreamGeometry(geometry, System.Windows.Media.Color.FromArgb(Canvas.CurrentColor.Value.A, Canvas.CurrentColor.Value.R, Canvas.CurrentColor.Value.G, Canvas.CurrentColor.Value.B));
                Canvas.Restore(id);
            }
        }
        public void DrawLine(float x1, float y1, float x2, float y2, float lineThickness, System.Drawing.Color? color)
        {
            Canvas.CurrentColor = defaultColor;
            Canvas.DrawLine(x1, y1, x2, y2, lineThickness);
        }
        public void DrawGlyphRunWithOffset
          (Display.AttributedGlyphRun<Fonts, Glyph> run, PointF offset, System.Drawing.Color? color)
        {
            if (color == null)
            {
                color = defaultColor;
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
            Guid id = Guid.NewGuid();
            Canvas.Save(id);
            Canvas.Translate(textPosition.X, textPosition.Y);
            Canvas.CurrentColor = color;
            foreach (var (glyph, kernAfter, foreground) in run.GlyphInfos)
            {
                var typeface = glyph.Typeface;
                var scale = typeface.CalculateScaleToPixelFromPointSize(pointSize);
                var index = glyph.Info.GlyphIndex;
                Canvas.CurrentColor = foreground ?? color;
                var wpfPath = new WpfPath(Canvas);
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
            Canvas.Restore(id);
        }

        public void FillRect(RectangleF rect, System.Drawing.Color color)
        {
            Canvas.CurrentColor = color;
            Canvas.FillRect(rect.X, rect.Y, rect.Width, rect.Height);
        }
        public void RestoreState(Guid id)
        {
            Canvas.Restore(id);
        }
        public void SaveState(Guid id)
        {
            Canvas.Save(id);
        }

        public void Translate(PointF dxy) => Canvas.Translate(dxy.X, dxy.Y);
    }
}
