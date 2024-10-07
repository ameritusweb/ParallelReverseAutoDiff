using Emgu.CV.Structure;
using Emgu.CV;

public static class CIELABGradientCalculator
{
    public class PolarGradientResult
    {
        public float Magnitude { get; set; }  // r
        public float AngleRadians { get; set; }  // θ in radians
        public string DirectionName { get; set; }  // Optional: for human-readable interpretation
    }

    private static readonly Dictionary<string, float[,]> Kernels = new Dictionary<string, float[,]>
    {
        { "N",  new float[,] {{ -1, -1, -1 }, { 0, 0, 0 }, { 1, 1, 1 }} },
        { "NE", new float[,] {{ -1, -1, 0 }, { -1, 0, 1 }, { 0, 1, 1 }} },
        { "E",  new float[,] {{ -1, 0, 1 }, { -1, 0, 1 }, { -1, 0, 1 }} },
        { "SE", new float[,] {{ 0, 1, 1 }, { -1, 0, 1 }, { -1, -1, 0 }} },
        { "S",  new float[,] {{ 1, 1, 1 }, { 0, 0, 0 }, { -1, -1, -1 }} },
        { "SW", new float[,] {{ 1, 1, 0 }, { 1, 0, -1 }, { 0, -1, -1 }} },
        { "W",  new float[,] {{ 1, 0, -1 }, { 1, 0, -1 }, { 1, 0, -1 }} },
        { "NW", new float[,] {{ 0, -1, -1 }, { 1, 0, -1 }, { 1, 1, 0 }} }
    };

    private static readonly Dictionary<string, int> DirectionToAngle = new Dictionary<string, int>
    {
        { "N", 90 }, { "NE", 45 }, { "E", 0 }, { "SE", 315 },
        { "S", 270 }, { "SW", 225 }, { "W", 180 }, { "NW", 135 }
    };

    /// <summary>
    /// Computes the LAB gradients in polar form (r, θ) for a given 3x3 LAB window.
    /// </summary>
    /// <param name="labWindow">A 3x3 window of the LAB image.</param>
    /// <returns>A dictionary of PolarGradientResult objects for L, a, and b channels.</returns>
    public static Dictionary<string, PolarGradientResult> CalculateLABGradientsPolar(Image<Lab, float> labWindow)
    {
        var results = new Dictionary<string, PolarGradientResult>();

        // Process each of the L, a, and b channels (0 = L, 1 = a, 2 = b)
        for (int i = 0; i < 3; i++)
        {
            var gradients = new Dictionary<string, float>();
            foreach (var kernel in Kernels)
            {
                float sum = 0;
                for (int x = 0; x < 3; x++)
                {
                    for (int y = 0; y < 3; y++)
                    {
                        sum += labWindow.Data[x, y, i] * kernel.Value[x, y];
                    }
                }
                gradients[kernel.Key] = sum;
            }

            // Find the direction with the maximum absolute gradient
            var maxDirection = gradients.OrderByDescending(kv => Math.Abs(kv.Value)).First().Key;
            var magnitude = gradients[maxDirection];
            var angleDegrees = DirectionToAngle[maxDirection];
            var angleRadians = (float)(angleDegrees * Math.PI / 180.0);  // Convert degrees to radians

            // Assign the result for L, a, or b
            results[i == 0 ? "L" : i == 1 ? "a" : "b"] = new PolarGradientResult
            {
                Magnitude = magnitude,  // r
                AngleRadians = angleRadians,  // θ in radians
                DirectionName = maxDirection  // Optional: human-readable direction name
            };
        }

        return results;
    }
}
