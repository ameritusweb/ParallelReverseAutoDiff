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
            Matrix dInput1 = new Matrix(input1.Rows, input1.Cols);
            Matrix dInput2 = new Matrix(input2.Rows, input2.Cols);
            Matrix dWeights = new Matrix(weights.Rows, weights.Cols);

            double sumX = summationX.Sum();
            double sumY = summationY.Sum();

            // Gradients for output magnitude
            double dResultMagnitude_dSumX = sumX / Math.Sqrt(sumX * sumX + sumY * sumY);
            double dResultMagnitude_dSumY = sumY / Math.Sqrt(sumX * sumX + sumY * sumY);

            // Gradients for output angle
            double dResultAngle_dSumX = -sumY / (Math.Pow(sumX, 2) + Math.Pow(sumY, 2));
            double dResultAngle_dSumY = sumX / (Math.Pow(sumX, 2) + Math.Pow(sumY, 2));

            double dMagnitudeOutput = dOutput[0, 0]; // Gradient of the loss function with respect to the output magnitude
            double dAngleOutput = dOutput[0, 1];     // Gradient of the loss function with respect to the output angle

            // Partial derivatives of atan2 for the aggregated sumX and sumY
            double dAtan2_dX = -sumY / (sumX * sumX + sumY * sumY);
            double dAtan2_dY = sumX / (sumX * sumX + sumY * sumY);

            // Updating gradients with respect to resultMagnitude and resultAngle
            for (int i = 0; i < input1.Rows; i++)
            {
                for (int j = 0; j < input1.Cols / 2; j++)
                {
                    double magnitude = input1[i, j];
                    double angle = input1[i, j + input1.Cols / 2];
                    double wMagnitude = input2[i, j];
                    double wAngle = input2[i, j + input2.Cols / 2];

                    double x1 = magnitude * Math.Cos(angle);
                    double y1 = magnitude * Math.Sin(angle);
                    double x2 = wMagnitude * Math.Cos(wAngle);
                    double y2 = wMagnitude * Math.Sin(wAngle);

                    double sumx = x1 + x2;
                    double sumy = y1 + y2;

                    // Compute derivatives for magnitude
                    double dResultMagnitudeLocal_dX1 = 2 * sumx * x1 * weights[i, j] * weights[i, j];
                    double dResultMagnitudeLocal_dY1 = 2 * sumy * y1 * weights[i, j] * weights[i, j];

                    double dResultMagnitudeLocal_dX2 = 2 * sumx * x2 * weights[i, j] * weights[i, j];
                    double dResultMagnitudeLocal_dY2 = 2 * sumy * y2 * weights[i, j] * weights[i, j];

                    // Compute derivatives for angle
                    double dResultAngle_dX1 = -dResultAngle_dSumX;
                    double dResultAngle_dY1 = dResultAngle_dSumY;

                    double dResultAngle_dX2 = -dResultAngle_dSumX;
                    double dResultAngle_dY2 = dResultAngle_dSumY;

                    // Apply chain rule to propagate back to input1, input2, and weights
                    dInput1[i, j] += dMagnitudeOutput * dResultMagnitude_dSumX * dResultMagnitudeLocal_dX1;
                    dInput1[i, j + input1.Cols / 2] += dMagnitudeOutput * dResultMagnitude_dSumY * dResultMagnitudeLocal_dY1;

                    dInput2[i, j] += dMagnitudeOutput * dResultMagnitude_dSumX * dResultMagnitudeLocal_dX2;
                    dInput2[i, j + input2.Cols / 2] += dMagnitudeOutput * dResultMagnitude_dSumY * dResultMagnitudeLocal_dY2;

                    dInput1[i, j] += dAngleOutput * dResultAngle_dX1;
                    dInput1[i, j + input1.Cols / 2] += dAngleOutput * dResultAngle_dY1;

                    dInput2[i, j] += dAngleOutput * dResultAngle_dX2;
                    dInput2[i, j + input2.Cols / 2] += dAngleOutput * dResultAngle_dY2;

                    // Compute how changes in weights affect sumX and sumY
                    double dSumX_dWeight = Math.Sqrt(sumx * sumx + sumy * sumy) * (2 * weights[i, j]);
                    double dSumY_dWeight = dSumX_dWeight; // Assuming similar relationship due to symmetry

                    // Refine derivative with respect to weights for angle
                    double dResultAngle_dWeight = dResultAngle_dSumX * dSumX_dWeight + dResultAngle_dSumY * dSumY_dWeight;

                    // Update dWeights with contributions from both magnitude and angle
                    dWeights[i, j] += dMagnitudeOutput * (2 * weights[i, j] * Math.Sqrt(sumx * sumx + sumy * sumy)); // Contribution from magnitude
                    dWeights[i, j] += dAngleOutput * dResultAngle_dWeight;                                         // Contribution from angle
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .AddInputGradient(dWeights)
                .Build();
        }
    }
}
