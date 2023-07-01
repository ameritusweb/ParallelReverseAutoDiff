//------------------------------------------------------------------------------
// <copyright file="DualWeightedOperation.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// A dual weighted transformation function.
    /// </summary>
    public class DualWeightedOperation : Operation
    {
        private Matrix input;
        private Matrix w1; // Weights Wj
        private Matrix w2; // Weights W2i

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new DualWeightedOperation();
        }

        /// <summary>
        /// Performs the forward operation for the dual weighted transformation function.
        /// </summary>
        /// <param name="input">The input to the operation.</param>
        /// <param name="w1">The first set of weights (Wj).</param>
        /// <param name="w2">The second set of weights (W2i).</param>
        /// <returns>The output of the operation.</returns>
        public Matrix Forward(Matrix input, Matrix w1, Matrix w2)
        {
            this.input = input;
            this.w1 = w1;
            this.w2 = w2;

            int numRows = input.Rows;
            int numCols = w1.Cols;

            this.Output = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    for (int k = 0; k < input.Cols; k++)
                    {
                        this.Output[i, j] += input[i, k] * w1[k, j] * w2[i, k];
                    }
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix gradOutput)
        {
            int numRows = this.input.Rows;
            int numCols = this.w1.Cols;

            Matrix gradInput = new Matrix(numRows, this.input.Cols);
            Matrix gradW1 = new Matrix(this.input.Cols, numCols);
            Matrix gradW2 = new Matrix(numRows, this.input.Cols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    for (int k = 0; k < this.input.Cols; k++)
                    {
                        gradW1[k, j] += gradOutput[i, j] * this.input[i, k] * this.w2[i, k];
                        gradW2[i, k] += gradOutput[i, j] * this.input[i, k] * this.w1[k, j];
                        gradInput[i, k] += gradOutput[i, j] * this.w1[k, j] * this.w2[i, k];
                    }
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(gradInput)
                .AddWeightGradient(gradW1)
                .AddWeightGradient(gradW2)
                .Build();
        }
    }
}