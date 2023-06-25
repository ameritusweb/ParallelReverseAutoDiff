//------------------------------------------------------------------------------
// <copyright file="BatchScaleAndShiftOperation.cs" author="ameritusweb" date="5/8/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Batch scale and shift operation.
    /// </summary>
    public class BatchScaleAndShiftOperation : BatchOperation
    {
        private ScaleAndShiftOperation[] operations;

        private BatchScaleAndShiftOperation(int batchSize)
        {
            this.operations = new ScaleAndShiftOperation[batchSize];
        }

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchScaleAndShiftOperation(net.Parameters.BatchSize);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.operations, (key, oldValue) => this.operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.operations = this.IntermediateOperationArrays[id].OfType<ScaleAndShiftOperation>().ToArray();
        }

        /// <summary>
        /// The forward pass of the scale and shift operation.
        /// </summary>
        /// <param name="input">The input matrix.</param>
        /// <param name="beta">The beta matrix.</param>
        /// <param name="gamma">The gamma matrix.</param>
        /// <returns>The output matrix.</returns>
        public DeepMatrix Forward(DeepMatrix input, Matrix beta, Matrix gamma)
        {
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.operations[i] = new ScaleAndShiftOperation();
                matrixArray[i] = this.operations[i].Forward(input[i], beta, gamma);
            });
            this.DeepOutput = new DeepMatrix(matrixArray);
            return this.DeepOutput;
        }

        /// <summary>
        /// Calculates the gradient of the scale and shift operation with respect to the input, beta, and gamma matrices.
        /// </summary>
        /// <param name="gradOutput">The gradient of the output matrix.</param>
        /// <returns>A tuple containing the gradients for the input, beta, and gamma matrices.</returns>
        public override BackwardResult[] Backward(DeepMatrix gradOutput)
        {
            var result = new BackwardResult[gradOutput.Depth];
            Parallel.For(0, gradOutput.Depth, i =>
            {
                result[i] = this.operations[i].Backward(gradOutput[i]);
            });
            return result;
        }
    }
}
