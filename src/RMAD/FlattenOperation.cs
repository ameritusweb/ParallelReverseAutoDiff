//------------------------------------------------------------------------------
// <copyright file="FlattenOperation.cs" author="ameritusweb" date="5/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Flatten operation for a DeepMatrix.
    /// </summary>
    public class FlattenOperation : Operation, IOperation
    {
        private DeepMatrix input;
        private int depth;
        private int rows;
        private int cols;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new FlattenOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            var data = new[] { (object)this.input, (object)this.depth, (object)this.rows, (object)this.cols };
            this.IntermediateObjectArrays.AddOrUpdate(id, data, (_, _) => data);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateObjectArrays[id];
            this.input = (DeepMatrix)restored[0];
            this.depth = (int)restored[1];
            this.rows = (int)restored[2];
            this.cols = (int)restored[3];
        }

        /// <summary>
        /// The forward pass of the flatten operation.
        /// </summary>
        /// <param name="input">The input matrix.</param>
        /// <returns>The output matrix.</returns>
        public Matrix Forward(DeepMatrix input)
        {
            this.input = input;
            this.depth = input.Depth;
            this.rows = input.Rows;
            this.cols = input.Cols;

            int totalElements = this.depth * this.rows * this.cols;

            this.Output = new Matrix(totalElements, 1);

            int index = 0;
            for (int d = 0; d < this.depth; d++)
            {
                for (int i = 0; i < this.rows; i++)
                {
                    for (int j = 0; j < this.cols; j++)
                    {
                        this.Output[index, 0] = input[d, i, j];
                        index++;
                    }
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            DeepMatrix dInput = new DeepMatrix(this.depth, this.rows, this.cols);

            int index = 0;
            for (int d = 0; d < this.depth; d++)
            {
                for (int i = 0; i < this.rows; i++)
                {
                    for (int j = 0; j < this.cols; j++)
                    {
                        dInput[d, i, j] = dOutput[index, 0];
                        index++;
                    }
                }
            }

            return new BackwardResultBuilder()
                .AddDeepInputGradient(dInput)
                .Build();
        }
    }
}