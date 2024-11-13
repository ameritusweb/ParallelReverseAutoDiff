using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Text.Json;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.Vision
{
    public class HSLValue
    {
        [JsonPropertyName("Item1")]
        public float Hue { get; set; }

        [JsonPropertyName("Item2")]
        public float Saturation { get; set; }

        [JsonPropertyName("Item3")]
        public float Lightness { get; set; }

        // Constructor for easy creation
        public HSLValue(float h, float s, float l)
        {
            Hue = h;
            Saturation = s;
            Lightness = l;
        }

        // Default constructor for JSON deserialization
        public HSLValue() { }
    }
}
