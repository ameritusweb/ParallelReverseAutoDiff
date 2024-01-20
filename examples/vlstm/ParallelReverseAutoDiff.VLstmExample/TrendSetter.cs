using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.VLstmExample
{
    public class TrendSetter
    {
        public void Set()
        {
            TrendModel model = new TrendModel(0.1d, 2 * Math.PI); // Example amplitude and frequency

            for (double t = 0; t <= 12; t += 0.5) // Assuming t is in months
            {
                double trendValue = model.Evaluate(t, 20, 10, 10); // Example trend values
                Console.WriteLine($"Month {t}: Trend Value = {trendValue}");
            }
        }
    }
}
