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
        /// Calculate the Pearson correlation coefficient loss.
        /// </summary>
        /// <param name="magnitudes">The magnitudes.</param>
        /// <param name="targetAngle">The target angles.</param>
        /// <param name="epsilon">An epsilon.</param>
        /// <returns>The correlation.</returns>
        public static (double[,] Correlations, Matrix Gradient, double Loss) PearsonCorrelationLoss(Matrix[] magnitudes, double[] targetAngle, double epsilon = 1e-8)
        {
            int timeSteps = magnitudes.Length;
            int numRows = magnitudes[0].Rows;
            int numColumns = magnitudes[0].Cols;

            double[,] correlations = new double[numRows, numColumns];
            double loss = 0.0d;

            Matrix gradients = CalculateGradients(magnitudes, targetAngle);

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
                    var corr = correlations[row, col];
                    var sign = Math.Sign(corr);
                    gradients[row, col] *= sign;

                    loss -= Math.Abs(corr);
                }
            }

            return (correlations, gradients, loss);
        }

        /// <summary>
        /// Calculate and accumulate the gradients.
        /// </summary>
        /// <param name="magnitudes">The magnitudes.</param>
        /// <param name="targetAngle">The target angles.</param>
        /// <param name="epsilon">An epsilon.</param>
        /// <returns>The gradients.</returns>
        public static Matrix CalculateGradients(Matrix[] magnitudes, double[] targetAngle, double epsilon = 1e-8)
        {
            int timeSteps = magnitudes.Length;
            int numRows = magnitudes[0].Rows;
            int numColumns = magnitudes[0].Cols;

            Matrix dMagnitudes = new Matrix(numRows, numColumns);
            Matrix dMagnitudes2 = new Matrix(numRows, numColumns);

            double meanTarget = targetAngle.Average();

            for (int col = 0; col < numColumns; col++)
            {
                for (int row = 0; row < numRows; row++)
                {
                    double[] data = new double[timeSteps];
                    for (int t = 0; t < timeSteps; t++)
                    {
                        data[t] = magnitudes[t][row, col];
                    }

                    double meanData = data.Average();
                    double covariance = 0, varianceData = 0;
                    double[] diffData = new double[timeSteps];
                    double[] diffTarget = new double[timeSteps];

                    for (int i = 0; i < data.Length; i++)
                    {
                        diffData[i] = data[i] - meanData;
                        diffTarget[i] = targetAngle[i] - meanTarget;

                        covariance += diffData[i] * diffTarget[i];
                        varianceData += diffData[i] * diffData[i];
                    }

                    double denominator = Math.Sqrt((varianceData * targetAngle.Length) + epsilon);

                    for (int t = 0; t < timeSteps; t++)
                    {
                        double gradCov = diffTarget[t];
                        double gradVar = 2 * diffData[t];

                        // Correct application of chain rule for the gradient of the denominator
                        double gradDenom = gradVar / (2 * denominator);  // Adjust to reflect the change in variance and its impact on the square root

                        // Correctly apply the quotient rule to compute the gradient
                        double gradient = ((gradCov * denominator) - (covariance * gradDenom)) / Math.Pow(denominator, 2);

                        if (t == timeSteps - 1)
                        {
                            dMagnitudes2[row, col] += gradient;  // Sum up the gradients from all correlations
                        }

                        if (t > 0)
                        {
                            dMagnitudes[row, col] += gradient;
                        }
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
