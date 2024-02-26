namespace ParallelReverseAutoDiff.GravNetExample
{
    using System;

    public static class InterpolationFactory
    {
        private static int granularity = 100000; // Adjustable based on precision requirements
        private static readonly double[] sineTable;
        private static readonly double[] cosineTable;
        // Extend with additional tables for conversions if necessary

        // Precomputed values for cartesian to polar conversion
        private static readonly double[,] precomputedR;
        private static readonly double[,] precomputedTheta;
        private static readonly int gridSize;
        private static double gridSpacing = 0.01d; // Define based on the desired precision and range
        private static int desiredCoverageRange = 100; // Define based on the expected range of x and y values

        static InterpolationFactory()
        {
            sineTable = new double[granularity + 1];
            cosineTable = new double[granularity + 1];
            for (int i = 0; i <= granularity; i++)
            {
                double theta = i * 2 * Math.PI / granularity;
                sineTable[i] = Math.Sin(theta);
                cosineTable[i] = Math.Cos(theta);
            }

            // Initialize precomputed R and Theta arrays
            // Ensure gridSize is set to cover the desired range with the current gridSpacing
            gridSize = (int)Math.Ceiling(desiredCoverageRange / gridSpacing);

            // Reinitialize precomputed arrays with new gridSize
            precomputedR = new double[gridSize, gridSize];
            precomputedTheta = new double[gridSize, gridSize];
        }

        public static void Initialize()
        {
            PrecomputeCartesianToPolar();
        }

        public static double InterpolateSine(double theta)
        {
            theta = NormalizeAngle(theta); // Ensure theta is within 0 to 2π
            int index = (int)(theta / (2 * Math.PI) * granularity);
            int nextIndex = (index + 1) % granularity;
            double fraction = (theta % (2 * Math.PI / granularity)) / (2 * Math.PI / granularity);
            return sineTable[index] + (sineTable[nextIndex] - sineTable[index]) * fraction;
        }

        public static double InterpolateCosine(double theta)
        {
            theta = NormalizeAngle(theta); // Ensure theta is within 0 to 2π
            int index = (int)(theta / (2 * Math.PI) * granularity);
            int nextIndex = (index + 1) % granularity;
            double fraction = (theta % (2 * Math.PI / granularity)) / (2 * Math.PI / granularity);
            return cosineTable[index] + (cosineTable[nextIndex] - cosineTable[index]) * fraction;
        }

        public static (double x, double y) PolarToCartesian(double r, double theta)
        {
            theta = NormalizeAngle(theta); // Ensure theta is within 0 to 2π
            double sinTheta = InterpolateSine(theta);
            double cosTheta = InterpolateCosine(theta);

            double x = r * cosTheta;
            double y = r * sinTheta;

            return (x, y);
        }

        private static double NormalizeAngle(double theta)
        {
            // Normalizes theta to be within the 0 to 2π range
            while (theta < 0) theta += 2 * Math.PI;
            while (theta >= 2 * Math.PI) theta -= 2 * Math.PI;
            return theta;
        }

        public static (double r, double theta) CartesianToPolar(double x, double y)
        {
            // Normalize x and y to fit within the grid
            double normalizedX = x + (gridSize * gridSpacing / 2); // Adjust if your grid is centered differently
            double normalizedY = y + (gridSize * gridSpacing / 2);

            // Calculate grid indices and fractional parts
            int ix = (int)(normalizedX / gridSpacing);
            int iy = (int)(normalizedY / gridSpacing);
            double fx = (normalizedX / gridSpacing) - ix;
            double fy = (normalizedY / gridSpacing) - iy;

            // Simple bounds checking for grid indices, considering bilinear interpolation needs
            ix = Math.Clamp(ix, 0, gridSize - 2);
            iy = Math.Clamp(iy, 0, gridSize - 2);

            // Bilinear interpolation for r
            double r00 = precomputedR[ix, iy];
            double r10 = precomputedR[ix + 1, iy];
            double r01 = precomputedR[ix, iy + 1];
            double r11 = precomputedR[ix + 1, iy + 1];
            double r1 = Interpolate(r00, r10, fx); // Interpolate along x
            double r2 = Interpolate(r01, r11, fx); // Interpolate along x for the next row
            double r = Interpolate(r1, r2, fy); // Interpolate the results along y

            // Lookup for theta
            double theta = precomputedTheta[ix, iy]; // Using the original, non-interpolated grid indices

            return (r, theta);
        }

        private static double Interpolate(double start, double end, double fraction)
        {
            return start + (end - start) * fraction;
        }

        private static void PrecomputeCartesianToPolar()
        {
            for (int i = 0; i < gridSize; i++)
            {
                for (int j = 0; j < gridSize; j++)
                {
                    double x = i * gridSpacing - (gridSize * gridSpacing / 2d); // Centering the grid
                    double y = j * gridSpacing - (gridSize * gridSpacing / 2d); // Centering the grid
                    precomputedR[i, j] = Math.Sqrt(x * x + y * y);
                    precomputedTheta[i, j] = Math.Atan2(y, x);
                }
            }
        }
    }
}
