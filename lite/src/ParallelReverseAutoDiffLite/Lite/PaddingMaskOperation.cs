//------------------------------------------------------------------------------
// <copyright file="PaddingMaskOperation.cs" author="ameritusweb" date="12/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// A padding mask operation for a graph attention network.
    /// </summary>
    public class PaddingMaskOperation : Operation
    {
        private const float MaskValue = -1e9f;  // Large negative number for masking
        private Matrix input; // NxM matrix
        private Matrix mask; // NxM mask

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static PaddingMaskOperation Instantiate(NeuralNetwork net)
        {
            return new PaddingMaskOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.input, this.mask }, (x, y) => new[] { this.input, this.mask });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateMatrixArrays[id];
            this.input = restored[0];
            this.mask = restored[1];
        }

        /// <summary>
        /// The forward pass of the padding mask operation.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="mask">The mask.</param>
        /// <returns>The masked input.</returns>
        public Matrix Forward(Matrix input, Matrix mask)
        {
            this.input = input;
            this.mask = mask;
            int numRows = input.Rows;
            int numCols = input.Cols;

            this.Output = new Matrix(numRows, numCols);

            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    this.Output[i][j] = mask[i][j] == 0.0d ? MaskValue : input[i][j];
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int numRows = dOutput.Rows;
            int numCols = dOutput.Cols;
            Matrix dInput = new Matrix(numRows, numCols);

            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    dInput[i][j] = this.mask[i][j] == 0.0d ? 0 : dOutput[i][j];
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .Build();
        }
    }
}
