//------------------------------------------------------------------------------
// <copyright file="DeepMatrixElementwiseAddOperation.cs" author="ameritusweb" date="9/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Custom operation for element-wise addition of DeepMatrix slices.
    /// </summary>
    public class DeepMatrixElementwiseAddOperation : Operation
    {
        private DeepMatrix input;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new DeepMatrixElementwiseAddOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            // Store the intermediate DeepMatrix
            this.IntermediateDeepMatrices.AddOrUpdate(id, this.input, (_, _) => this.input);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            // Restore the intermediate DeepMatrix
            this.input = this.IntermediateDeepMatrices[id];
        }

        /// <summary>
        /// Performs the forward operation.
        /// </summary>
        /// <param name="input">DeepMatrix of MxN matrices.</param>
        /// <returns>Single MxN matrix.</returns>
        public Matrix Forward(DeepMatrix input)
        {
            this.input = input;
            int rows = input.Rows;
            int cols = input.Cols;
            this.Output = new Matrix(rows, cols);

            foreach (Matrix slice in input)
            {
                this.Output += slice;  // Assuming that the += operator performs element-wise addition
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            DeepMatrix dInput = new DeepMatrix(this.input.Depth, this.input.Rows, this.input.Cols);

            // Since addition is element-wise and each element contributes linearly to the output,
            // the gradient is the same for all slices.
            for (int k = 0; k < this.input.Depth; k++)
            {
                dInput[k] = dOutput;
            }

            return new BackwardResultBuilder()
                .AddInputGradientArray(dInput)
                .Build();
        }
    }
}
