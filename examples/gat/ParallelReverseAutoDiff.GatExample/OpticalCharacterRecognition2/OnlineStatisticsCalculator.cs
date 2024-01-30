using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GatExample.OpticalCharacterRecognition2
{
    using System;

    public class OnlineStatisticsCalculator
    {
        private long count;
        private double mean;
        private double variance;
        private double m2; // Used for variance calculation
        private double min;
        private double max;

        public OnlineStatisticsCalculator()
        {
            Reset();
        }

        public void AddDataPoint(double x)
        {
            count++;

            // Update min and max
            if (count == 1)
            {
                min = max = x;
            }
            else
            {
                if (x < min)
                {
                    min = x;
                }

                if (x > max)
                {
                    max = x;
                }
            }

            // Calculate new mean
            double delta = x - mean;
            mean += delta / count;

            // Calculate new variance using Welford's method
            double delta2 = x - mean;
            m2 += delta * delta2;

            // Update variance
            if (count < 2)
            {
                variance = 0;
            }
            else
            {
                variance = m2 / (count - 1);
            }
        }

        public void AddDataPoints(double[] dataPoints)
        {
            foreach (double x in dataPoints)
            {
                AddDataPoint(x);
            }
        }

        public double GetMean()
        {
            return mean;
        }

        public double GetVariance()
        {
            return variance;
        }

        public double GetStandardDeviation()
        {
            return Math.Sqrt(variance);
        }

        public double GetMin()
        {
            return min;
        }

        public double GetMax()
        {
            return max;
        }

        public void Reset()
        {
            count = 0;
            mean = 0;
            variance = 0;
            m2 = 0;
            min = double.MaxValue;
            max = double.MinValue;
        }
    }
}
