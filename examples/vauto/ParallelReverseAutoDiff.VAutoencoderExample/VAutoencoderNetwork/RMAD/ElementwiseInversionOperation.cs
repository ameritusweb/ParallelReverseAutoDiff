//------------------------------------------------------------------------------
// <copyright file="ElementwiseInversionOperation.cs" author="ameritusweb" date="4/8/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise inversion operation.
    /// </summary>
    public class ElementwiseInversionOperation : Operation
    {
        private Matrix input;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseInversionOperation();
        }

        /// <summary>
        /// Performs the forward operation for the element-wise inversion function.
        /// </summary>
        /// <param name="input">The input to the element-wise inversion operation.</param>
        /// <returns>The output of the element-wise inversion operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;

            this.Output = new Matrix(input.Rows, input.Cols);
            Parallel.For(0, input.Rows, i =>
            {
                for (int j = 0; j < input.Cols; j++)
                {
                    this.Output[i, j] = 1 - input[i, j];
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInput = new Matrix(this.input.Rows, this.input.Cols);

            // The derivative of (1 - x) with respect to x is -1 for every element.
            Parallel.For(0, this.input.Rows, i =>
            {
                for (int j = 0; j < this.input.Cols; j++)
                {
                    dInput[i, j] = -dOutput[i, j];
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .Build();
        }
    }
}
