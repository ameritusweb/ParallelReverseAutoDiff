using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.Vision
{
    public class CenterRegion
    {
        [JsonPropertyName("values")]
        public float[][] Values { get; set; }  // Changed from float[][][] to float[][]

        [JsonPropertyName("average")]
        public float[] Average { get; set; }  // Changed from HSLValue to float[]

        [JsonPropertyName("x")]
        public int X { get; set; }

        [JsonPropertyName("y")]
        public int Y { get; set; }

        [JsonPropertyName("width")]
        public int Width { get; set; }

        [JsonPropertyName("height")]
        public int Height { get; set; }
    }
}
