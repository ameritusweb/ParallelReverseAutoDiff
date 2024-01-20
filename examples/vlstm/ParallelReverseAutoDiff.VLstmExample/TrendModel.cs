using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.VLstmExample
{
    public class TrendModel
    {
        private SineWaveTransformer transformer;

        public TrendModel(double amplitude, double frequency)
        {
            transformer = new SineWaveTransformer(amplitude, frequency);
        }

        public double Evaluate(double t, double yearlyIncrease, double semiAnnualDecrease, double monthlyIncrease)
        {
            // Composite function for rotation angle, with subtle influence
            double angleRadians = CalculateCompositeRotationAngle(t, yearlyIncrease, semiAnnualDecrease, monthlyIncrease);

            Console.WriteLine($"Month {t}: Rotation Angle = {angleRadians}");

            transformer.UpdateRotationAngle(angleRadians);

            // Transform point
            (double _, double yPrime) = transformer.TransformPoint(t);
            return yPrime;
        }

        private double CalculateCompositeRotationAngle(double t, double yearlyIncrease, double semiAnnualDecrease, double monthlyIncrease)
        {
            // Subtle adjustments to rotation angle
            double angleAdjustmentFactor = 0.05; // Adjust this factor as needed

            double yearlyComponent = yearlyIncrease * Math.Sin(2 * Math.PI * t / 12.0);
            double semiAnnualComponent = semiAnnualDecrease * Math.Sin(2 * Math.PI * t / 6.0);
            double monthlyComponent = monthlyIncrease * Math.Sin(2 * Math.PI * t);

            Console.WriteLine($"Month {t}: Yearly Component = {yearlyComponent}, SemiAnnual Component = {semiAnnualComponent}, Monthly Component = {monthlyComponent}");

            // Calculate a subtle rotation angle
            return (yearlyComponent + semiAnnualComponent + monthlyComponent) * angleAdjustmentFactor;
        }
    }
}
