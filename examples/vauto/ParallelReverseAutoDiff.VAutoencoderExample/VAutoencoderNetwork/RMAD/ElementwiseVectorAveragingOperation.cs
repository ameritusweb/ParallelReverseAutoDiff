//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorAveragingOperation.cs" author="ameritusweb" date="4/8/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise vector averaging operation.
    /// </summary>
    public class ElementwiseVectorAveragingOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseVectorAveragingOperation();
        }

        /// <summary>
        /// Performs the forward operation for the element-wise vector scaling function.
        /// </summary>
        /// <param name="input1">The first input to the element-wise vector scaling operation.</param>
        /// <param name="input2">The second input to the element-wise vector scaling operation.</param>
        /// <returns>The output of the element-wise vector scaling operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2)
        {
            this.input1 = input1;
            this.input2 = input2;

            this.Output = new Matrix(this.input1.Rows, this.input1.Cols);
            Parallel.For(0, input1.Rows, i =>
            {
                for (int j = 0; j < input1.Cols / 2; j++)
                {
                    double magnitude = input1[i, j];
                    double angle = input1[i, j + (input1.Cols / 2)];

                    double wMagnitude = input2[i, j];
                    double wAngle = input2[i, j + (input2.Cols / 2)];

                    double avgMagnitude = (magnitude + wMagnitude) / 2;
                    double avgAngle = (angle + wAngle) / 2;

                    this.Output[i, j] = avgMagnitude;
                    this.Output[i, j + (this.input1.Cols / 2)] = avgAngle;
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInput1 = new Matrix(this.input1.Rows, this.input1.Cols);
            Matrix dInput2 = new Matrix(this.input2.Rows, this.input2.Cols);

            Parallel.For(0, this.input1.Rows, i =>
            {
                for (int j = 0; j < this.input1.Cols / 2; j++)
                {
                    // Since the operation is averaging, the gradient of each input is simply 0.5 * gradient of the output.
                    double dMagnitude = dOutput[i, j] * 0.5;
                    double dAngle = dOutput[i, j + (this.input1.Cols / 2)] * 0.5;

                    // The gradients for both inputs are the same as they equally contribute to the average.
                    dInput1[i, j] = dMagnitude;
                    dInput1[i, j + (this.input1.Cols / 2)] = dAngle;

                    dInput2[i, j] = dMagnitude;
                    dInput2[i, j + (this.input2.Cols / 2)] = dAngle;
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .Build();
        }
    }
}
