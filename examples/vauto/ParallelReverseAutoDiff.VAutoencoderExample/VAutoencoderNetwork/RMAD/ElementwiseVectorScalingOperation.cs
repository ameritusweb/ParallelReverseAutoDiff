//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorScalingOperation.cs" author="ameritusweb" date="4/8/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise vector scaling operation.
    /// </summary>
    public class ElementwiseVectorScalingOperation : Operation
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
            return new ElementwiseVectorScalingOperation();
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

                    double wScaling = input2[i, j];

                    this.Output[i, j] = magnitude * wScaling;
                    this.Output[i, j + (this.input1.Cols / 2)] = angle;
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
                    var magnitude = this.input1[i, j];
                    var angle = this.input1[i, j + (this.input1.Cols / 2)];

                    var wScaling = this.input2[i, j];

                    dInput1[i, j] = dOutput[i, j] * wScaling;

                    dInput2[i, j] = dOutput[i, j] * magnitude;
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .Build();
        }
    }
}
