﻿//------------------------------------------------------------------------------
// <copyright file="BatchMatrixBroadcastOperation.cs" author="ameritusweb" date="6/16/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Batch matrix broadcast operation.
    /// </summary>
    public class BatchMatrixBroadcastOperation : BatchOperation
    {
        private MatrixBroadcastOperation[] operations;

        private BatchMatrixBroadcastOperation(int batchSize)
        {
            this.operations = new MatrixBroadcastOperation[batchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchMatrixBroadcastOperation(net.Parameters.BatchSize);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.operations, (key, oldValue) => this.operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.operations = this.IntermediateOperationArrays[id].OfType<MatrixBroadcastOperation>().ToArray();
        }

        /// <summary>
        /// Performs the forward operation for the matrix broadcast function.
        /// </summary>
        /// <param name="input">A matrix to broadcast.</param>
        /// <param name="targetRows">The target number of rows.</param>
        /// <param name="targetCols">The target number of columns.</param>
        /// <returns>The output of the matrix broadcast operation.</returns>
        public DeepMatrix Forward(DeepMatrix input, int[] targetRows, int[] targetCols)
        {
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.operations[i] = new MatrixBroadcastOperation();
                matrixArray[i] = this.operations[i].Forward(input[i], targetRows[i], targetCols[i]);
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