//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorContributionOperation.cs" author="ameritusweb" date="4/8/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise vector contribution operation.
    /// </summary>
    public class ElementwiseVectorContributionOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static ElementwiseVectorContributionOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseVectorContributionOperation();
        }

        /// <summary>
        /// Performs the forward operation for the element-wise vector contribution function.
        /// </summary>
        /// <param name="input1">The first input to the element-wise vector contribution operation.</param>
        /// <param name="input2">The second input to the element-wise vector contribution operation.</param>
        /// <returns>The output of the element-wise vector contribution operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2)
        {
            this.input1 = input1;
            this.input2 = input2;

            this.Output = new Matrix(1, 2);
            var vectorA = new Matrix(1, 2);
            var vectorB = new Matrix(1, 2);
            vectorA[0, 1] = Math.PI / 4;
            vectorB[0, 1] = 3 * Math.PI / 4;
            for (int i = 0; i < input1.Rows; ++i)
            {
                for (int j = 0; j < input1.Cols / 2; j++)
                {
                    double magnitude = input1[i, j];
                    double angle = input1[i, j + (input1.Cols / 2)];
                    if (input2[i, j] > 0.5d)
                    {
                        vectorA[0, 0] += magnitude + angle;
                    }
                    else
                    {
                        vectorB[0, 0] += magnitude + angle;
                    }
                }
            }

            this.Output[0, 0] = (vectorA[0, 0] * Math.Cos(vectorA[0, 1])) + (vectorB[0, 0] * Math.Cos(vectorB[0, 1]));
            this.Output[0, 1] = (vectorA[0, 0] * Math.Sin(vectorA[0, 1])) + (vectorB[0, 0] * Math.Sin(vectorB[0, 1]));

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInput1 = new Matrix(this.input1.Rows, this.input1.Cols);

            double dOutX = dOutput[0, 0];  // Derivative from the output's X component
            double dOutY = dOutput[0, 1];  // Derivative from the output's Y component

            for (int i = 0; i < this.input1.Rows; i++)
            {
                for (int j = 0; j < this.input1.Cols / 2; j++)
                {
                    double angle = this.input2[i, j] > 0.5 ? Math.PI / 4 : 3 * Math.PI / 4;
                    double cosAngle = Math.Cos(angle);
                    double sinAngle = Math.Sin(angle);

                    // Compute the gradient for each input element based on its contribution to the output
                    var commonGradient = (dOutX * cosAngle) + (dOutY * sinAngle);
                    dInput1[i, j] = commonGradient;
                    dInput1[i, j + (this.input1.Cols / 2)] = commonGradient;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .Build();
        }
    }
}
