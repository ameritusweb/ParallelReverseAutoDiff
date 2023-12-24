//------------------------------------------------------------------------------
// <copyright file="DeepReLUOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Performs the forward and backward operations for the ReLU activation function.
    /// </summary>
    public class DeepReLUOperation : DeepOperation
    {
        private DeepMatrix input;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IDeepOperation Instantiate(NeuralNetwork net)
        {
            return new DeepReLUOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateDeepMatrices.AddOrUpdate(id, this.input, (x, y) => this.input);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.input = this.IntermediateDeepMatrices[id];
        }

        /// <summary>
        /// Performs the forward operation for the ReLU activation function.
        /// </summary>
        /// <param name="input">The input to the ReLU operation.</param>
        /// <returns>The output of the ReLU operation.</returns>
        public DeepMatrix Forward(DeepMatrix input)
        {
            this.input = input;
            int depth = input.Depth;
            int rows = input.Rows;
            int cols = input.Cols;
            this.DeepOutput = new DeepMatrix(depth, rows, cols);

            Parallel.For(0, depth, d =>
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        double x = input[d, i, j];
                        this.DeepOutput[d, i, j] = x > 0 ? x : 0;
                    }
                }
            });

            return this.DeepOutput;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(DeepMatrix dLdOutput)
        {
            int depth = dLdOutput.Depth;
            int rows = dLdOutput.Rows;
            int cols = dLdOutput.Cols;
            DeepMatrix dLdInput = new DeepMatrix(depth, rows, cols);

            Parallel.For(0, depth, d =>
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        double x = this.input[d, i, j];
                        double gradient = x > 0 ? 1.0 : 0.0;
                        dLdInput[d, i, j] = dLdOutput[d, i, j] * gradient;
                    }
                }
            });

            return new BackwardResultBuilder()
                .AddDeepInputGradient(dLdInput)
                .Build();
        }
    }
}
