using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.Vision
{
    public class HSLImageData
    {
        [JsonPropertyName("width")]
        public int Width { get; set; }

        [JsonPropertyName("height")]
        public int Height { get; set; }

        [JsonPropertyName("hsl_grid")]
        public float[][][] HSLGrid { get; set; }

        [JsonPropertyName("center_region")]
        public CenterRegion CenterRegion { get; set; }

        // Helper method to get HSL value at specific coordinates
        public HSLValue GetHSLAt(int x, int y)
        {
            if (x < 0 || x >= Width || y < 0 || y >= Height)
                throw new ArgumentOutOfRangeException($"Coordinates ({x},{y}) out of range for {Width}x{Height} image");

            var hslArray = HSLGrid[y][x];
            return new HSLValue(hslArray[0], hslArray[1], hslArray[2]);
        }

        // Helper method to check if a coordinate is in the masked center region
        public bool IsInCenterRegion(int x, int y)
        {
            return x >= CenterRegion.X &&
                   x < (CenterRegion.X + CenterRegion.Width) &&
                   y >= CenterRegion.Y &&
                   y < (CenterRegion.Y + CenterRegion.Height);
        }

        // Helper method to get original center region value at specific coordinates
        public HSLValue GetOriginalCenterValueAt(int x, int y)
        {
            if (!IsInCenterRegion(x, y))
                throw new ArgumentException($"Coordinates ({x},{y}) not in center region");

            int localX = x - CenterRegion.X;
            int localY = y - CenterRegion.Y;
            int index = localY * CenterRegion.Width + localX;  // Convert 2D position to 1D array index
            var values = CenterRegion.Values[index];  // Get the [h,s,l] array for this position
            return new HSLValue(values[0], values[1], values[2]);
        }

        // Helper method to get the average HSL value of the center region
        public HSLValue GetCenterRegionAverage()
        {
            var avg = CenterRegion.Average;
            return new HSLValue(avg[0], avg[1], avg[2]);
        }

        public float[] Extract(HSLComponent component)
        {
            float[] result = new float[Width * Height];

            for (int y = 0; y < Height; y++)
            {
                for (int x = 0; x < Width; x++)
                {
                    int index = y * Width + x;
                    var hslArray = HSLGrid[y][x];
                    result[index] = hslArray[(int)component];
                }
            }

            return result;
        }
    }
}
