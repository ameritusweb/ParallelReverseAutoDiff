//------------------------------------------------------------------------------
// <copyright file="BatchMatrixSumOperation.cs" author="ameritusweb" date="6/16/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Batch matrix sum operation.
    /// </summary>
    public class BatchMatrixSumOperation : BatchOperation
    {
        private MatrixSumOperation[] operations;

        private BatchMatrixSumOperation(int batchSize)
        {
            this.operations = new MatrixSumOperation[batchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchMatrixSumOperation(net.Parameters.BatchSize);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.operations, (key, oldValue) => this.operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.operations = this.IntermediateOperationArrays[id].OfType<MatrixSumOperation>().ToArray();
        }

        /// <summary>
        /// Performs the forward operation for the matrix sum function.
        /// </summary>
        /// <param name="input">The input to the matrix sum operation, an NxM matrix.</param>
        /// <returns>The output of the matrix sum operation, an Nx1 matrix.</returns>
        public DeepMatrix Forward(DeepMatrix input)
        {
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.operations[i] = new MatrixSumOperation();
                matrixArray[i] = this.operations[i].Forward(input[i]);
            });
            this.DeepOutput = new DeepMatrix(matrixArray);
            return this.DeepOutput;
        }

        /// <inheritdoc />
        public override BackwardResult[] Backward(DeepMatrix dOutput)
        {
            var result = new BackwardResult[dOutput.Depth];
            Parallel.For(0, dOutput.Depth, i =>
            {
                result[i] = this.operations[i].Backward(dOutput[i]);
            });
            return result;
        }
    }
}
