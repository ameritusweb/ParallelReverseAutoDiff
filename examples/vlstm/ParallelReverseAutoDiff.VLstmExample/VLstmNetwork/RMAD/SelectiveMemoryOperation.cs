//------------------------------------------------------------------------------
// <copyright file="SelectiveMemoryOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Performs the forward and backward operations for the element-wise square function.
    /// </summary>
    public class SelectiveMemoryOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;
        private Matrix sigmoid;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new SelectiveMemoryOperation();
        }

        /// <summary>
        /// Performs the forward operation for the element-wise square function.
        /// </summary>
        /// <param name="input1">The input to the element-wise square operation.</param>
        /// <param name="input2">The input to the element-wise square operation.</param>
        /// <param name="sigmoid">The input to the element-wise square operation.</param>
        /// <returns>The output of the element-wise square operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix sigmoid)
        {
            this.input1 = input1;
            this.input2 = input2;
            this.sigmoid = sigmoid;
            int rows = input1.Length;
            int cols = input1[0].Length;
            this.Output = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double x = input1[i][j];
                    this.Output[i][j] = x * x;
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int rows = dLdOutput.Length;
            int cols = dLdOutput[0].Length;
            Matrix dLdInput = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double x = this.input1[i][j];
                    double gradient = 2 * x;
                    dLdInput[i][j] = dLdOutput[i][j] * gradient;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }
    }
}
