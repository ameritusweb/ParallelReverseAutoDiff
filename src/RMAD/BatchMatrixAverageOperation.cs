﻿//------------------------------------------------------------------------------
// <copyright file="BatchMatrixAverageOperation.cs" author="ameritusweb" date="6/16/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Batch matrix average operation.
    /// </summary>
    public class BatchMatrixAverageOperation : BatchOperation
    {
        private MatrixAverageOperation[] operations;

        private BatchMatrixAverageOperation(int batchSize)
        {
            this.operations = new MatrixAverageOperation[batchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchMatrixAverageOperation(net.Parameters.BatchSize);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.operations, (key, oldValue) => this.operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.operations = this.IntermediateOperationArrays[id].OfType<MatrixAverageOperation>().ToArray();
        }

        /// <summary>
        /// Performs the forward operation for the matrix average function.
        /// </summary>
        /// <param name="input">The input to the matrix average operation.</param>
        /// <returns>The output of the matrix average operation.</returns>
        public DeepMatrix Forward(DeepMatrix input)
        {
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.operations[i] = new MatrixAverageOperation();
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