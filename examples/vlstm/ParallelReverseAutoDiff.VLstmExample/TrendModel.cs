//------------------------------------------------------------------------------
// <copyright file="TrendModel.cs" author="ameritusweb" date="1/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.VLstmExample
{
    /// <summary>
    /// A trend model.
    /// </summary>
    public class TrendModel
    {
        private readonly ISineWaveTransformer mainTransformer;
        private readonly ISineWaveTransformer yearlyTransformer;
        private readonly ISineWaveTransformer semiAnnualTransformer;
        private readonly ISineWaveTransformer monthlyTransformer;

        /// <summary>
        /// Initializes a new instance of the <see cref="TrendModel"/> class.
        /// </summary>
        /// <param name="main">The main sine wave transformer.</param>
        /// <param name="yearly">The yearly sine wave transformer.</param>
        /// <param name="semiAnnual">The semi-annual sine wave transformer.</param>
        /// <param name="monthly">The monthly sine wave transformer.</param>
        public TrendModel(ISineWaveTransformer main, ISineWaveTransformer yearly, ISineWaveTransformer semiAnnual, ISineWaveTransformer monthly)
        {
            this.mainTransformer = main;
            this.yearlyTransformer = yearly;
            this.semiAnnualTransformer = semiAnnual;
            this.monthlyTransformer = monthly;
        }

        /// <summary>
        /// Evaluates the trend model.
        /// </summary>
        /// <param name="t">The time step.</param>
        /// <param name="yearlyIncrease">The yearly increase.</param>
        /// <param name="semiAnnualDecrease">The semi-annual decrease.</param>
        /// <param name="monthlyIncrease">The monthly increase.</param>
        /// <returns>The value.</returns>
        public double Evaluate(double t, double yearlyIncrease, double semiAnnualDecrease, double monthlyIncrease)
        {
            this.UpdateTransformers(yearlyIncrease, semiAnnualDecrease, monthlyIncrease);

            double angleRadians = this.CalculateCompositeRotationAngle(t);
            Console.WriteLine($"Month {t}: Rotation Angle = {angleRadians}");

            this.mainTransformer.UpdateRotationAngle(angleRadians);
            return this.mainTransformer.Transform(t);
        }

        private static double ToRadians(double degrees) => degrees * Math.PI / 180;

        private void UpdateTransformers(double yearlyIncrease, double semiAnnualDecrease, double monthlyIncrease)
        {
            this.yearlyTransformer.UpdateRotationAngle(ToRadians(yearlyIncrease));
            this.semiAnnualTransformer.UpdateRotationAngle(ToRadians(semiAnnualDecrease));
            this.monthlyTransformer.UpdateRotationAngle(ToRadians(monthlyIncrease));
        }

        private double CalculateCompositeRotationAngle(double t) =>
            this.yearlyTransformer.Transform(t) + this.semiAnnualTransformer.Transform(t) + this.monthlyTransformer.Transform(t);
    }
}
