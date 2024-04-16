//------------------------------------------------------------------------------
// <copyright file="CorrelationCalculator2.cs" author="ameritusweb" date="5/7/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A gated recurrent network correlation calculator.
    /// </summary>
    public class CorrelationCalculator2
    {
        /// <summary>
        /// Calculate the Pearson correlation coefficient loss.
        /// </summary>
        /// <param name="magnitudes">The magnitudes.</param>
        /// <param name="targetAngle">The target angles.</param>
        /// <param name="epsilon">An epsilon.</param>
        /// <returns>The correlation.</returns>
        public static (double[] Correlations, Matrix Gradient, double Loss) PearsonCorrelationLoss(Matrix[] magnitudes, double[] targetAngle, double epsilon = 1e-8)
        {
            int timeSteps = magnitudes.Length;
            int numRows = magnitudes[0].Rows;

            double[] correlations = new double[numRows];
            double loss = 0.0;

            Matrix gradients = CalculateGradientsForSummedRows(magnitudes, targetAngle);

            for (int row = 0; row < numRows; row++)
            {
                double[] summedRowData = new double[timeSteps];
                for (int t = 0; t < timeSteps; t++)
                {
                    summedRowData[t] = magnitudes[t][row].Sum();
                }

                correlations[row] = CalculateSinglePearson(summedRowData, targetAngle, epsilon);

                // Subtracting because we are minimizing the negative of absolute values of correlations
                loss -= Math.Abs(correlations[row]);
            }

            return (correlations, gradients, loss);
        }

        /// <summary>
        /// Calculate gradients for summed rows.
        /// </summary>
        /// <param name="magnitudes">The magnitudes.</param>
        /// <param name="targetAngle">The target angles.</param>
        /// <param name="epsilon">An epsilon.</param>
        /// <returns>The gradients.</returns>
        public static Matrix CalculateGradientsForSummedRows(Matrix[] magnitudes, double[] targetAngle, double epsilon = 1e-8)
        {
            int timeSteps = magnitudes.Length;
            int numRows = magnitudes[0].Rows;
            Matrix dMagnitudes = new Matrix(numRows, magnitudes[0].Cols);
            Matrix dd = new Matrix(numRows, magnitudes[0].Cols);

            double meanTarget = targetAngle.Average();

            for (int row = 0; row < numRows; row++)
            {
                double[] summedRowData = new double[timeSteps];
                for (int t = 0; t < timeSteps; t++)
                {
                    summedRowData[t] = magnitudes[t][row].Sum();
                }

                double meanData = summedRowData.Average();
                double varianceData = 0, covariance = 0;

                for (int i = 0; i < timeSteps; i++)
                {
                    varianceData += Math.Pow(summedRowData[i] - meanData, 2);
                    covariance += (summedRowData[i] - meanData) * (targetAngle[i] - meanTarget);
                }

                double denominator = Math.Sqrt((varianceData * targetAngle.Length) + epsilon);

                for (int t = 0; t < timeSteps; t++)
                {
                    for (int col = 0; col < magnitudes[t].Cols; col++)
                    {
                        double partialSum = magnitudes[t][row, col];
                        double gradCov = targetAngle[t] - meanTarget;
                        double gradVar = 2 * (partialSum - meanData);

                        double gradDenom = gradVar / (2 * denominator);
                        double gradient = ((gradCov * denominator) - (covariance * gradDenom)) / Math.Pow(denominator, 2);

                        if (t > 0)
                        {
                            dMagnitudes[row, col] += gradient;
                        }

                        dd[row, col] += gradient;
                    }
                }
            }

            return dMagnitudes;
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
