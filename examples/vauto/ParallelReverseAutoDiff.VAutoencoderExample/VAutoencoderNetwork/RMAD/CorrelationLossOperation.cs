//------------------------------------------------------------------------------
// <copyright file="CorrelationLossOperation.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.VGruExample.VGruNetwork.RMAD
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Correlation loss operation.
    /// </summary>
    public class CorrelationLossOperation
    {
        private Matrix[] predictionsOverTime;
        private double[] targetAnglesOverTime;
        private int numTimeSteps;
        private CalculatedValues calculatedValues;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static CorrelationLossOperation Instantiate(NeuralNetwork net)
        {
            return new CorrelationLossOperation();
        }

        /// <summary>
        /// Performs the forward operation for the correlation loss function.
        /// </summary>
        /// <param name="predictionsOverTime">The predictions matrix.</param>
        /// <param name="targetAnglesOverTime">The target angles.</param>
        /// <returns>The scalar loss value.</returns>
        public Matrix Forward(Matrix[] predictionsOverTime, double[] targetAnglesOverTime)
        {
            this.predictionsOverTime = predictionsOverTime;
            this.targetAnglesOverTime = targetAnglesOverTime;
            this.numTimeSteps = predictionsOverTime.Length;
            var loss = 0d;
            var rows = this.predictionsOverTime[0].Rows;
            int cols = this.predictionsOverTime[0].Cols;
            Matrix[] dPredictions = new Matrix[this.numTimeSteps];
            Matrix aggregated = new Matrix(rows, cols);

            for (int k = 0; k < this.numTimeSteps; k++)
            {
                dPredictions[k] = new Matrix(rows, cols); // Initialize matrices to store gradients
            }

            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    double[] predictionOverTime = new double[this.numTimeSteps];
                    for (int k = 0; k < this.numTimeSteps; k++)
                    {
                        predictionOverTime[k] = this.predictionsOverTime[k][i, j];
                    }

                    var lossAnalyzed = this.AnalyzePredictionsOverTime(predictionOverTime, this.targetAnglesOverTime, out double[] gradients);
                    var adjustedLoss = Math.Pow(1 + lossAnalyzed, 2d);
                    loss += adjustedLoss;

                    // Adjust gradients according to the chain rule
                    double gradientFactor = 2 * (1 + lossAnalyzed);
                    for (int k = 0; k < this.numTimeSteps; k++)
                    {
                        dPredictions[k][i, j] += gradients[k] * gradientFactor; // Summing adjusted gradients for each element
                        aggregated[i, j] += gradients[k] * gradientFactor; // Summing adjusted gradients for each element
                    }
                }
            }

            this.calculatedValues = new CalculatedValues { Gradients = dPredictions, Aggregated = aggregated };
            var output = new Matrix(1, 1);
            output[0, 0] = loss;

            return output;
        }

        /// <summary>
        /// Runs the backward operation for the correlation loss function.
        /// </summary>
        /// <returns>The gradient with respect to the predictions.</returns>
        public Matrix Backward()
        {
            return this.calculatedValues.Aggregated;
        }

        private double AnalyzePredictionsOverTime(double[] data, double[] targetAngle, out double[] gradients)
        {
            if (data.Length != targetAngle.Length)
            {
                throw new ArgumentException("Data and targetAngle arrays must have the same length.");
            }

            double meanData = data.Average();
            double meanTarget = targetAngle.Average();
            double stdData = Math.Sqrt(data.Select(d => (d - meanData) * (d - meanData)).Average());
            double[] dStdData_dData = this.CalculateStdDevGradient(data);
            double stdTarget = Math.Sqrt(targetAngle.Select(t => (t - meanTarget) * (t - meanTarget)).Average());

            double loss = 0;
            gradients = new double[data.Length];

            for (int i = 1; i < data.Length; i++)
            {
                double u = data[i] - data[i - 1];
                double v = stdData;
                double dataChange = u / v;
                double targetChange = (targetAngle[i] - targetAngle[i - 1]) / stdTarget;

                // Calculate deviation for the loss based on direction alignment
                double deviation = Math.Abs(dataChange * targetChange);
                bool correctDirection = (dataChange > 0 && targetChange > 0) || (dataChange < 0 && targetChange < 0);

                if (correctDirection)
                {
                    loss -= deviation;  // Rewarding correct alignment by subtracting deviation from loss
                }
                else
                {
                    loss += deviation;  // Penalizing wrong direction by adding deviation to loss
                }

                // Gradient calculation with direction consideration
                double dLoss_dDataChange = targetChange * Math.Sign(dataChange * targetChange);
                if (correctDirection)
                {
                    dLoss_dDataChange *= -1;  // Invert gradient effect to decrease parameter adjustments
                }

                double uPrime_i = 1;  // derivative of u with respect to data[i]
                double uPrime_iMinus1 = -1;  // derivative of u with respect to data[i-1]
                double vPrime_i = dStdData_dData[i];
                double vPrime_iMinus1 = dStdData_dData[i - 1];

                gradients[i] -= dLoss_dDataChange * ((uPrime_i * v) - (u * vPrime_i)) / (v * v);
                gradients[i - 1] += dLoss_dDataChange * ((uPrime_iMinus1 * v) - (u * vPrime_iMinus1)) / (v * v);
            }

            return loss;
        }

        private double[] CalculateStdDevGradient(double[] data)
        {
            int n = data.Length;
            double mean = data.Average();
            double variance = data.Select(x => (x - mean) * (x - mean)).Sum() / n;
            double stdDev = Math.Sqrt(variance);

            double[] gradients = new double[n];
            for (int k = 0; k < n; k++)
            {
                // Applying the derived formula for the gradient of the standard deviation
                gradients[k] = (data[k] - mean) / (n * stdDev);
            }

            return gradients;
        }

        private struct CalculatedValues
        {
            public Matrix[] Gradients { get; internal set; }

            public Matrix Aggregated { get; internal set; }
        }
    }
}
