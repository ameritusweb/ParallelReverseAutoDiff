//------------------------------------------------------------------------------
// <copyright file="CorrelationCalculator.cs" author="ameritusweb" date="5/7/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A gated recurrent network correlation calculator.
    /// </summary>
    public class CorrelationCalculator
    {
        /// <summary>
        /// Calculate the Pearson correlation coefficient.
        /// </summary>
        /// <param name="magnitudes">The magnitudes.</param>
        /// <param name="targetAngle">The target angles.</param>
        /// <param name="epsilon">An epsilon.</param>
        /// <returns>The correlation.</returns>
        public static double[,] PearsonCorrelation(Matrix[] magnitudes, double[] targetAngle, double epsilon = 1e-8)
        {
            int timeSteps = magnitudes.Length;
            int numRows = magnitudes[0].Rows;
            int numColumns = magnitudes[0].Cols;

            double[,] correlations = new double[numRows, numColumns];

            for (int col = 0; col < numColumns; col++)
            {
                for (int row = 0; row < numRows; row++)
                {
                    double[] extractedData = new double[timeSteps];
                    for (int t = 0; t < timeSteps; t++)
                    {
                        extractedData[t] = magnitudes[t][row, col];
                    }

                    correlations[row, col] = CalculateSinglePearson(extractedData, targetAngle, epsilon);
                }
            }

            return correlations;
        }

        private static double CalculateSinglePearson(double[] data, double[] targetAngle, double epsilon)
        {
            double meanData = data.Average();
            double meanTarget = targetAngle.Average();

            double covariance = 0, varianceData = 0, varianceTarget = 0;

            for (int i = 0; i < data.Length; i++)
            {
                double diffData = data[i] - meanData;
                double diffTarget = targetAngle[i] - meanTarget;

                covariance += diffData * diffTarget;
                varianceData += diffData * diffData;
                varianceTarget += diffTarget * diffTarget;
            }

            double denominator = Math.Sqrt((varianceData * varianceTarget) + epsilon);
            return covariance / denominator;
        }
    }
}
