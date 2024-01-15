using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GravNetExample.VectorNetwork
{
    public class VectorVisualizer
    {
        private const int BitmapWidth = 4000;  // Adjust as needed
        private const int BitmapHeight = 7000; // Adjust as needed
        private const float scaleFactor = 500f; // Adjust as needed
        private Bitmap bitmap;
        private Graphics graphics;

        public VectorVisualizer()
        {
            this.bitmap = new Bitmap(BitmapWidth, BitmapHeight);
            this.graphics = Graphics.FromImage(bitmap);
            this.graphics.Clear(Color.White); // Set background color
        }

        public void Draw(double[,] vectors, string id)
        {
            PointF startPoint = new PointF(2000f, 6500f); // Starting point for the first vector

            for (int i = 0; i < vectors.GetLength(0); i++)
            {
                double[] vector = new double[2];
                vector[0] = vectors[i, 0];
                vector[1] = vectors[i, 1];
                startPoint = DrawVector(startPoint, vector[0], vector[1]);
            }

            SaveBitmap($"E:\\vnnstore\\vector-{id}.png");
        }

        private void DrawGradientLine(PointF start, PointF end)
        {
            const int numSegments = 100; // Number of segments for the gradient
            PointF[] points = new PointF[numSegments + 1];
            for (int i = 0; i <= numSegments; i++)
            {
                float t = (float)i / numSegments;
                points[i] = new PointF(Lerp(start.X, end.X, t), Lerp(start.Y, end.Y, t));
                using (var pen = new Pen(GradientColor(t), 2))
                {
                    if (i > 0)
                    {
                        this.graphics.DrawLine(pen, points[i - 1], points[i]);
                    }
                }
            }
        }

        private float Lerp(float start, float end, float t)
        {
            return start + (end - start) * t;
        }

        private Color GradientColor(float t)
        {
            // Linear interpolation for grayscale
            int colorValue = 250 - (int)(200 * t); // From gray 250 to gray 50
            return Color.FromArgb(colorValue, colorValue, colorValue);
        }

        public void SaveBitmap(string filePath)
        {
            this.bitmap.Save(filePath, ImageFormat.Png);
        }

        public PointF DrawVector(PointF start, double magnitude, double angle)
        {
            // Apply scaling to magnitude and starting point
            magnitude *= scaleFactor;
            PointF scaledStart = new PointF(start.X, start.Y);

            // Calculate the end point of the vector
            float endX = scaledStart.X + (float)(magnitude * Math.Cos(angle));
            float endY = scaledStart.Y - (float)(magnitude * Math.Sin(angle));
            PointF end = new PointF(endX, endY);

            // Draw the gradient line
            DrawGradientLine(scaledStart, end);

            // Return the original end point (not scaled) for chaining vectors
            return new PointF(start.X + (float)(magnitude * Math.Cos(angle)), start.Y - (float)(magnitude * Math.Sin(angle)));
        }
    }
}
