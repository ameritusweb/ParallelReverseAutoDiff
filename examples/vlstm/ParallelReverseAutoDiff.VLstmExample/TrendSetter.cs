//------------------------------------------------------------------------------
// <copyright file="TrendSetter.cs" author="ameritusweb" date="1/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.VLstmExample
{
    /// <summary>
    /// A trend setter.
    /// </summary>
    public class TrendSetter
    {
        private TrendModel model;

        /// <summary>
        /// Sets the trend.
        /// </summary>
        public void Set()
        {
            var mainTransformer = new SineWaveTransformer(5d, 2 * Math.PI);
            var yearlyTransformer = new SineWaveTransformer(1, 2 * Math.PI / 12); // Yearly trend
            var semiAnnualTransformer = new SineWaveTransformer(1, 2 * Math.PI / 6); // Semi-annual trend
            var monthlyTransformer = new SineWaveTransformer(1, 2 * Math.PI); // Monthly trend
            TrendModel model = new TrendModel(mainTransformer, yearlyTransformer, semiAnnualTransformer, monthlyTransformer); // Example amplitude and frequency
            this.model = model;

            double startValue = 100d; // Example starting value

            for (double t = 0; t <= 12; t += 0.25) // Assuming t is in months
            {
                double trendValue = model.Evaluate(t, 20, -10, 10); // Yearly increase, semi-annual decrease, monthly increase
                Console.WriteLine($"Month {t}: Change = {trendValue}");
                startValue += trendValue;
                Console.WriteLine($"Month {t}: Value = {startValue}");
            }
        }
    }
}
