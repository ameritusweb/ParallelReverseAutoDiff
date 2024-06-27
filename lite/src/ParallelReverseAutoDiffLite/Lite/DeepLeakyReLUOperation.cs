//------------------------------------------------------------------------------
// <copyright file="DeepLeakyReLUOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// A leaky ReLU operation.
    /// </summary>
    public class DeepLeakyReLUOperation : DeepOperation
    {
        private readonly float alpha;
        private DeepMatrix input;

        /// <summary>
        /// Initializes a new instance of the <see cref="DeepLeakyReLUOperation"/> class.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        public DeepLeakyReLUOperation(float alpha)
        {
            this.alpha = alpha;
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IDeepOperation Instantiate(NeuralNetwork net)
        {
            return new DeepLeakyReLUOperation(net.Parameters.LeakyReLUAlpha);
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
        /// The forward pass of the leaky ReLU operation.
        /// </summary>
        /// <param name="input">The input for the leaky ReLU operation.</param>
        /// <returns>The output for the leaky ReLU operation.</returns>
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
                        float x = input[d, i, j];
                        this.DeepOutput[d, i, j] = x > 0 ? x : this.alpha * x;
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
                        float x = this.input[d, i, j];
                        float gradient = x > 0 ? 1.0f : this.alpha;
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
