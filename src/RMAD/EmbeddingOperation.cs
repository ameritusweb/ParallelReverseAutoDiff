//------------------------------------------------------------------------------
// <copyright file="EmbeddingOperation.cs" author="ameritusweb" date="5/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// An embedding operation.
    /// </summary>
    public class EmbeddingOperation : Operation
    {
        private Matrix input;
        private Matrix weights;  // This is your embedding matrix

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new EmbeddingOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.input, this.weights, this.Output }, (_, _) => new[] { this.input, this.weights, this.Output });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateMatrixArrays[id];
            this.input = restored[0];
            this.weights = restored[1];
            this.Output = restored[2];
        }

        /// <summary>
        /// The forward pass of the embedding operation.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="weights">The weights.</param>
        /// <returns>The mebedded input.</returns>
        public Matrix Forward(Matrix input, Matrix weights)
        {
            this.input = input;
            this.weights = weights;
            this.Output = new Matrix(input.Rows, weights.Cols);

            for (int i = 0; i < input.Rows; i++)
            {
                int inputIdx = (int)input[i, 0];
                this.Output[i] = weights[inputIdx];  // Assign the corresponding embedding to the output
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dWeights = new Matrix(this.weights.Rows, this.weights.Cols);

            for (int i = 0; i < this.input.Rows; i++)
            {
                int inputIdx = (int)this.input[i, 0];
                for (int j = 0; j < dWeights.Cols; j++)
                {
                    dWeights[inputIdx, j] += dOutput[i, j];  // Update the gradients for the corresponding embedding
                }
            }

            // In embedding layer, the input gradients do not exist, as inputs are not trainable
            // Hence, they are set to null or an appropriately sized zero matrix
            Matrix dInput = new Matrix(this.input.Rows, this.input.Cols);

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .AddWeightGradient(dWeights)
                .Build();
        }
    }
}