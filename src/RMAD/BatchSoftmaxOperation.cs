//------------------------------------------------------------------------------
// <copyright file="BatchSoftmaxOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Batch softmax operation.
    /// </summary>
    public class BatchSoftmaxOperation : BatchOperation
    {
        private SoftmaxOperation[] operations;

        private BatchSoftmaxOperation(int batchSize)
        {
            this.operations = new SoftmaxOperation[batchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchSoftmaxOperation(net.Parameters.BatchSize);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.operations, (key, oldValue) => this.operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.operations = this.IntermediateOperationArrays[id].OfType<SoftmaxOperation>().ToArray();
        }

        /// <summary>
        /// Performs the forward operation for the softmax function.
        /// </summary>
        /// <param name="input">The input to the softmax operation.</param>
        /// <returns>The output of the softmax operation.</returns>
        public DeepMatrix Forward(DeepMatrix input)
        {
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.operations[i] = new SoftmaxOperation();
                matrixArray[i] = this.operations[i].Forward(input[i]);
            });
            this.DeepOutput = new DeepMatrix(matrixArray);
            return this.DeepOutput;
        }

        /// <inheritdoc />
        public override BackwardResult[] Backward(DeepMatrix dLdOutput)
        {
            var result = new BackwardResult[dLdOutput.Depth];
            Parallel.For(0, dLdOutput.Depth, i =>
            {
                result[i] = this.operations[i].Backward(dLdOutput[i]);
            });
            return result;
        }
    }
}
