//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorSummationOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise summation operation.
    /// </summary>
    public class ElementwiseVectorSummationOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;
        private Matrix weights;
        private double[] summationX;
        private double[] summationY;
        private CalculatedValues[,] calculatedValues;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseVectorSummationOperation();
        }

        /// <summary>
        /// Performs the forward operation for the element-wise vector summation function.
        /// </summary>
        /// <param name="input1">The first input to the element-wise vector summation operation.</param>
        /// <param name="input2">The second input to the element-wise vector summation operation.</param>
        /// <param name="weights">The weights.</param>
        /// <returns>The output of the element-wise vector summation operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights)
        {
            this.input1 = input1;
            this.input2 = input2;
            this.weights = weights;

            this.Output = new Matrix(1, 2);

            this.calculatedValues = new CalculatedValues[this.input1.Rows, this.input1.Cols / 2];

            double[] summationX = new double[input1.Rows];
            double[] summationY = new double[input1.Rows];
            Parallel.For(0, input1.Rows, i =>
            {
                double sumX = 0.0d;
                double sumY = 0.0d;
                for (int j = 0; j < input1.Cols / 2; j++)
                {
                    // Accessing the magnitudes and angles from the concatenated matrices
                    double magnitude = input1[i, j];
                    double angle = input1[i, j + (input1.Cols / 2)];

                    double wMagnitude = input2[i, j];
                    double wAngle = input2[i, j + (input2.Cols / 2)];

                    // Compute vector components
                    double x1 = magnitude * Math.Cos(angle);
                    double y1 = magnitude * Math.Sin(angle);
                    double x2 = wMagnitude * Math.Cos(wAngle);
                    double y2 = wMagnitude * Math.Sin(wAngle);

                    double sumx = x1 + x2;
                    double sumy = y1 + y2;

                    // Compute resultant vector magnitude and angle
                    double resultMagnitude = Math.Sqrt((sumx * sumx) + (sumy * sumy)) * weights[i, j];
                    double resultAngle = Math.Atan2(sumy, sumx);

                    sumX += resultMagnitude * Math.Cos(resultAngle);
                    sumY += resultMagnitude * Math.Sin(resultAngle);

                    this.CalculateAndStoreValues(i, j);
                }

                summationX[i] = sumX;
                summationY[i] = sumY;
            });

            this.summationX = summationX;
            this.summationY = summationY;

            double resultMagnitude = Math.Sqrt((summationX.Sum() * summationX.Sum()) + (summationY.Sum() * summationY.Sum()));
            double resultAngle = Math.Atan2(summationY.Sum(), summationX.Sum());

            this.Output[0, 0] = resultMagnitude;
            this.Output[0, 1] = resultAngle;

            return Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInput1 = new Matrix(input1.Rows, input1.Cols);
            Matrix dInput2 = new Matrix(input2.Rows, input2.Cols);
            Matrix dWeights = new Matrix(weights.Rows, weights.Cols);

            double sumX = summationX.Sum();
            double sumY = summationY.Sum();

            double sumXSquared = sumX * sumX;
            double sumYSquared = sumY * sumY;

            double combinedSquares = sumXSquared + sumYSquared;

            // Gradients for output magnitude
            double dResultMagnitude_dSumX = sumX / Math.Sqrt(combinedSquares);
            double dResultMagnitude_dSumY = sumY / Math.Sqrt(combinedSquares);

            // Gradients for output angle
            double dResultAngle_dSumX = -sumY / combinedSquares;
            double dResultAngle_dSumY = sumX / combinedSquares;

            double dMagnitudeOutput = dOutput[0, 0]; // Gradient of the loss function with respect to the output magnitude
            double dAngleOutput = dOutput[0, 1];     // Gradient of the loss function with respect to the output angle

            // Partial derivatives of atan2 for the aggregated sumX and sumY
            double dAtan2_dX = dResultAngle_dSumX;
            double dAtan2_dY = dResultAngle_dSumY;

            // Updating gradients with respect to resultMagnitude and resultAngle
            Parallel.For(0, this.input1.Rows, i =>
            {
                for (int j = 0; j < this.input1.Cols / 2; j++)
                {
                    // Compute derivatives for angle
                    double dResultAngle_dX1 = -dResultAngle_dSumX;
                    double dResultAngle_dY1 = dResultAngle_dSumY;

                    double dResultAngle_dX2 = -dResultAngle_dSumX;
                    double dResultAngle_dY2 = dResultAngle_dSumY;

                    // Apply chain rule to propagate back to input1, input2, and weights
                    dInput1[i, j] += (dMagnitudeOutput * dResultMagnitude_dSumX * this.calculatedValues[i, j].DResultMagnitudeLocalDX1) + (dAngleOutput * dResultAngle_dX1);
                    dInput1[i, j + (this.input1.Cols / 2)] += (dMagnitudeOutput * dResultMagnitude_dSumY * this.calculatedValues[i, j].DResultMagnitudeLocalDY1) + (dAngleOutput * dResultAngle_dY1);

                    dInput2[i, j] += (dMagnitudeOutput * dResultMagnitude_dSumX * this.calculatedValues[i, j].DResultMagnitudeLocalDX2) + (dAngleOutput * dResultAngle_dX2);
                    dInput2[i, j + (this.input2.Cols / 2)] += (dMagnitudeOutput * dResultMagnitude_dSumY * this.calculatedValues[i, j].DResultMagnitudeLocalDY2) + (dAngleOutput * dResultAngle_dY2);

                    // Apply the chain rule to calculate the gradient of resultAngle with respect to the weight
                    double dResultAngle_dWeightX = dAtan2_dX * this.calculatedValues[i, j].DSumXDWeight;
                    double dResultAngle_dWeightY = dAtan2_dY * this.calculatedValues[i, j].DSumYDWeight;
                    double dResultAngle_dWeight = dResultAngle_dWeightX + dResultAngle_dWeightY;

                    // Update dWeights with contributions from both magnitude and angle
                    dWeights[i, j] += dMagnitudeOutput * (this.weights[i, j] * this.calculatedValues[i, j].CombinedMagnitude); // Contribution from magnitude
                    dWeights[i, j] += dAngleOutput * dResultAngle_dWeight;                                     // Contribution from angle
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .AddInputGradient(dWeights)
                .Build();
        }

        private void CalculateAndStoreValues(int i, int j)
        {
            // Calculate and store values
            CalculatedValues values;
            var magnitude = this.input1[i, j];
            var angle = this.input1[i, j + (this.input1.Cols / 2)];
            var wMagnitude = this.input2[i, j];
            var wAngle = this.input2[i, j + (this.input2.Cols / 2)];

            var x1 = magnitude * Math.Cos(angle);
            var y1 = magnitude * Math.Sin(angle);
            var x2 = wMagnitude * Math.Cos(wAngle);
            var y2 = wMagnitude * Math.Sin(wAngle);

            var combinedX = x1 + x2;
            var combinedY = y1 + y2;

            values.CombinedMagnitude = Math.Sqrt((combinedX * combinedX) + (combinedY * combinedY));
            values.DResultMagnitudeLocalDX1 = combinedX * x1 * this.weights[i, j];
            values.DResultMagnitudeLocalDY1 = combinedY * y1 * this.weights[i, j];
            values.DResultMagnitudeLocalDX2 = combinedX * x2 * this.weights[i, j];
            values.DResultMagnitudeLocalDY2 = combinedY * y2 * this.weights[i, j];
            values.DSumXDWeight = this.weights[i, j] * combinedX * values.CombinedMagnitude;
            values.DSumYDWeight = this.weights[i, j] * combinedY * values.CombinedMagnitude;

            this.calculatedValues[i, j] = values;
        }

        private struct CalculatedValues
        {
            public double CombinedMagnitude;

            public double DResultMagnitudeLocalDX1;
            public double DResultMagnitudeLocalDY1;

            public double DResultMagnitudeLocalDX2;
            public double DResultMagnitudeLocalDY2;

            public double DSumXDWeight;
            public double DSumYDWeight;
        }
    }
}
