using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.VLstmExample
{
    public class SineWaveTransformer
    {
        public double Amplitude { get; set; }
        public double Frequency { get; set; }
        public double RotationAngleRadians { get; set; }

        public SineWaveTransformer(double amplitude, double frequency)
        {
            Amplitude = amplitude;
            Frequency = frequency;
        }

        public (double, double) TransformPoint(double x)
        {
            double y = Amplitude * Math.Sin(Frequency * x);
            double xPrime = x * Math.Cos(RotationAngleRadians) - y * Math.Sin(RotationAngleRadians);
            double yPrime = x * Math.Sin(RotationAngleRadians) + y * Math.Cos(RotationAngleRadians);
            return (xPrime, yPrime);
        }

        public void UpdateRotationAngle(double angleRadians)
        {
            RotationAngleRadians = angleRadians;
        }

        public double Transform(double x)
        {
            double y = Amplitude * Math.Sin(Frequency * x + RotationAngleRadians);
            return y;
        }
    }
}
