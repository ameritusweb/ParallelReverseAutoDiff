//------------------------------------------------------------------------------
// <copyright file="BatchAmplifiedSigmoidOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// The Batch sigmoid operation utilizing gradient amplification.
    /// </summary>
    public class BatchAmplifiedSigmoidOperation : BatchOperation
    {
        private AmplifiedSigmoidOperation[] operations;

        private BatchAmplifiedSigmoidOperation(int batchSize)
        {
            this.operations = new AmplifiedSigmoidOperation[batchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchAmplifiedSigmoidOperation(net.Parameters.BatchSize);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.operations, (key, oldValue) => this.operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.operations = this.IntermediateOperationArrays[id].OfType<AmplifiedSigmoidOperation>().ToArray();
        }

        /// <summary>
        /// The forward pass of the operation.
        /// </summary>
        /// <param name="input">The input for the operation.</param>
        /// <returns>The output for the operation.</returns>
        public DeepMatrix Forward(DeepMatrix input)
        {
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.operations[i] = new AmplifiedSigmoidOperation();
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
