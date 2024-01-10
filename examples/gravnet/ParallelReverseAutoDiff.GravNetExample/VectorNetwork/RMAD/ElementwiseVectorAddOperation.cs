//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorAddOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise add operation.
    /// </summary>
    public class ElementwiseVectorAddOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;
        private Matrix weights;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseVectorAddOperation();
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
            // Assuminginput1 and input2 have the same dimensions
            var Output = new Matrix(input1.Rows, input1.Cols);

            Parallel.For(0, input1.Rows, i =>
            {
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

                    // Store results in the output matrix
                    Output[i, j] = resultMagnitude;
                    Output[i, j + input1.Cols / 2] = resultAngle;
                }
            });

            return Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dMagnitudesAngles = new Matrix(this.input1.Rows, this.input1.Cols);
            Matrix dWMagnitudesAngles = new Matrix(this.input2.Rows, this.input2.Cols);
            Matrix dWeights = new Matrix(weights.Rows, weights.Cols);

            Parallel.For(0, this.input1.Rows, i =>
            {
                for (int j = 0; j < this.input1.Cols / 2; j++)
                {
                    // Recompute relevant forward pass values
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

                    double resultMagnitude = Math.Sqrt(sumx * sumx + sumy * sumy) * this.weights[i, j] * this.weights[i, j];

                    // Gradient for magnitude and weighted magnitude
                    double dResultMagnitude_dMagnitude = (sumx * Math.Cos(angle) + sumy * Math.Sin(angle)) / resultMagnitude;
                    double dResultMagnitude_dWMagnitude = (sumx * Math.Cos(wAngle) + sumy * Math.Sin(wAngle)) / resultMagnitude;
                    dMagnitudesAngles[i, j] += dOutput[i, j] * dResultMagnitude_dMagnitude;
                    dWMagnitudesAngles[i, j] += dOutput[i, j] * dResultMagnitude_dWMagnitude;

                    // Gradient for angle and weighted angle
                    double dResultMagnitude_dAngle = -magnitude * (sumx * Math.Sin(angle) - sumy * Math.Cos(angle)) / resultMagnitude;
                    double dResultMagnitude_dWAngle = -wMagnitude * (sumx * Math.Sin(wAngle) - sumy * Math.Cos(wAngle)) / resultMagnitude;
                    dMagnitudesAngles[i, j + this.input1.Cols / 2] += dOutput[i, j + this.input1.Cols / 2] * dResultMagnitude_dAngle;
                    dWMagnitudesAngles[i, j + this.input2.Cols / 2] += dOutput[i, j + this.input2.Cols / 2] * dResultMagnitude_dWAngle;

                    // Gradient for weights
                    double dResultMagnitude_dWeight = resultMagnitude * 2 * weights[i, j];
                    dWeights[i, j] += dOutput[i, j] * dResultMagnitude_dWeight;
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dMagnitudesAngles)
                .AddInputGradient(dWMagnitudesAngles)
                .AddInputGradient(dWeights)
                .Build();
        }
    }
}
