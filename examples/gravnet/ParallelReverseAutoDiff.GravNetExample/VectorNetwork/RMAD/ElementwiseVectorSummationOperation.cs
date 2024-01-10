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
        /// Performs the forward operation for the Hadamard product function.
        /// </summary>
        /// <param name="input1">The first input to the Hadamard product operation.</param>
        /// <param name="input2">The second input to the Hadamard product operation.</param>
        /// <returns>The output of the Hadamard product operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights)
        {
            this.input1 = input1;
            this.input2 = input2;
            this.weights = weights;

            this.Output = new Matrix(1, 2);

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
                    double angle = input1[i, j + input1.Cols / 2];

                    double wMagnitude = input2[i, j];
                    double wAngle = input2[i, j + input2.Cols / 2];

                    // Compute vector components
                    double x1 = magnitude * Math.Cos(angle);
                    double y1 = magnitude * Math.Sin(angle);
                    double x2 = wMagnitude * Math.Cos(wAngle);
                    double y2 = wMagnitude * Math.Sin(wAngle);

                    double sumx = x1 + x2;
                    double sumy = y1 + y2;

                    // Compute resultant vector magnitude and angle
                    double resultMagnitude = Math.Sqrt(sumx * sumx + sumy * sumy) * weights[i, j] * weights[i, j];
                    double resultAngle = Math.Atan2(sumy, sumx);

                    sumX += resultMagnitude * Math.Cos(resultAngle);
                    sumY += resultMagnitude * Math.Sin(resultAngle);
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
            Matrix dInput1 = new Matrix(this.input1.Rows, this.input1.Cols);
            Matrix dInput2 = new Matrix(this.input2.Rows, this.input2.Cols);
            Matrix dWeights = new Matrix(this.weights.Rows, this.weights.Cols);

            double sumX = this.summationX.Sum();
            double sumY = this.summationY.Sum();

            double resultMagnitude = Math.Sqrt((sumX * sumX) + (sumY * sumY));
            double resultAngle = Math.Atan2(sumY, sumX);

            double dResultMagnitude_dSumX = sumX / resultMagnitude;
            double dResultMagnitude_dSumY = sumY / resultMagnitude;

            Parallel.For(0, this.input1.Rows, i =>
            {
                double sumXLocal = this.summationX[i];
                double sumYLocal = this.summationY[i];

                for (int j = 0; j < this.input1.Cols / 2; j++)
                {
                    double magnitude = this.input1[i, j];
                    double angle = this.input1[i, j + this.input1.Cols / 2];
                    double wMagnitude = this.input2[i, j];
                    double wAngle = this.input2[i, j + this.input2.Cols / 2];

                    double x1 = magnitude * Math.Cos(angle);
                    double y1 = magnitude * Math.Sin(angle);
                    double x2 = wMagnitude * Math.Cos(wAngle);
                    double y2 = wMagnitude * Math.Sin(wAngle);

                    double sumx = x1 + x2;
                    double sumy = y1 + y2;

                    double resultMagnitudeLocal = Math.Sqrt(sumx * sumx + sumy * sumy) * this.weights[i, j] * this.weights[i, j];
                    double resultAngleLocal = Math.Atan2(sumy, sumx);

                    double dResultMagnitudeLocal_dX1 = 2 * sumx * x1 * this.weights[i, j] * this.weights[i, j];
                    double dResultMagnitudeLocal_dY1 = 2 * sumy * y1 * this.weights[i, j] * this.weights[i, j];

                    double dResultMagnitudeLocal_dX2 = 2 * sumx * x2 * this.weights[i, j] * this.weights[i, j];
                    double dResultMagnitudeLocal_dY2 = 2 * sumy * y2 * this.weights[i, j] * this.weights[i, j];

                    double dResultMagnitudeLocal_dWeight = 2 * resultMagnitudeLocal * this.weights[i, j];

                    // Apply chain rule for each component
                    dInput1[i, j] += dOutput[0, 0] * dResultMagnitude_dSumX * dResultMagnitudeLocal_dX1;
                    dInput1[i, j + input1.Cols / 2] += dOutput[0, 0] * dResultMagnitude_dSumY * dResultMagnitudeLocal_dY1;

                    dInput2[i, j] += dOutput[0, 0] * dResultMagnitude_dSumX * dResultMagnitudeLocal_dX2;
                    dInput2[i, j + input2.Cols / 2] += dOutput[0, 0] * dResultMagnitude_dSumY * dResultMagnitudeLocal_dY2;

                    dWeights[i, j] += dOutput[0, 0] * dResultMagnitudeLocal_dWeight;
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .AddInputGradient(dWeights)
                .Build();
        }
    }
}
