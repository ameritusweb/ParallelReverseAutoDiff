//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorMultiplyOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise multiplication operation.
    /// </summary>
    public class ElementwiseVectorMultiplyOperation : Operation
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
            return new ElementwiseVectorMultiplyOperation();
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
            var Output = new Matrix(input1.Rows,input1.Cols);

            Parallel.For(0,input1.Rows, i =>
            {
                for (int j = 0; j <input1.Cols / 2; j++)
                {
                    // Accessing the magnitudes and angles from the concatenated matrices
                    double magnitude =input1[i, j];
                    double angle =input1[i, j + input1.Cols / 2];

                    double wMagnitude = input2[i, j];
                    double wAngle = input2[i, j + input2.Cols / 2];

                    // Compute vector components
                    double x1 = magnitude * Math.Cos(angle);
                    double y1 = magnitude * Math.Sin(angle);
                    double x2 = wMagnitude * Math.Cos(wAngle);
                    double y2 = wMagnitude * Math.Sin(wAngle);

                    // Select vector direction based on weight
                    double deltax = weights[i, j] > 0 ? x2 - x1 : x1 - x2;
                    double deltay = weights[i, j] > 0 ? y2 - y1 : y1 - y2;

                    // Compute resultant vector magnitude and angle
                    double resultMagnitude = Math.Sqrt(deltax * deltax + deltay * deltay) * Math.Pow(weights[i, j], 2d);
                    double resultAngle = Math.Atan2(deltay, deltax);

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
                    double deltax = weights[i, j] > 0 ? x2 - x1 : x1 - x2;
                    double deltay = weights[i, j] > 0 ? y2 - y1 : y1 - y2;

                    double resultMagnitude = Math.Sqrt(deltax * deltax + deltay * deltay) * Math.Pow(this.weights[i, j], 2d);

                    // Gradient for magnitude
                    double dResultMagnitude_dMagnitude = (deltax * Math.Cos(angle) + deltay * Math.Sin(angle)) / resultMagnitude;
                    double dResultMagnitude_dWMagnitude = (deltax * Math.Cos(wAngle) + deltay * Math.Sin(wAngle)) / resultMagnitude;
                    dMagnitudesAngles[i, j] += dOutput[i, j] * dResultMagnitude_dMagnitude;
                    dWMagnitudesAngles[i, j] += dOutput[i, j] * dResultMagnitude_dWMagnitude;

                    // Gradient for angle
                    double dResultMagnitude_dAngle = -magnitude * ((deltax * Math.Sin(angle) - deltay * Math.Cos(angle)) / resultMagnitude);
                    double dResultMagnitude_dWAngle = -wMagnitude * ((deltax * Math.Sin(wAngle) - deltay * Math.Cos(wAngle)) / resultMagnitude);
                    dMagnitudesAngles[i, j + this.input1.Cols / 2] += dOutput[i, j + this.input1.Cols / 2] * dResultMagnitude_dAngle;
                    dWMagnitudesAngles[i, j + this.input2.Cols / 2] += dOutput[i, j + this.input1.Cols / 2] * dResultMagnitude_dWAngle;

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
