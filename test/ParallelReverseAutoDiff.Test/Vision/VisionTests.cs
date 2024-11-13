using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace ParallelReverseAutoDiff.Test.Vision
{
    public class VisionTests
    {
        [Fact]
        public void TestVision()
        {
            string assemblyPath = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            string path = Path.Combine(assemblyPath, "Vision\\JSON", "hsl.json");
            var imageData = HSLImageLoader.LoadFromFile(path);
            var hues = imageData.Extract(HSLComponent.Hue);
            var sats = imageData.Extract(HSLComponent.Saturation);
            var lights = imageData.Extract(HSLComponent.Lightness);


        }
    }
}
