//------------------------------------------------------------------------------
// <copyright file="BatchLayerNormalizationOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Batch layer normalization operation.
    /// </summary>
    public class BatchLayerNormalizationOperation : BatchOperation
    {
        private LayerNormalizationOperation[] operations;

        private BatchLayerNormalizationOperation(int batchSize)
        {
            this.operations = new LayerNormalizationOperation[batchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchLayerNormalizationOperation(net.Parameters.BatchSize);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.operations, (key, oldValue) => this.operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.operations = this.IntermediateOperationArrays[id].OfType<LayerNormalizationOperation>().ToArray();
        }

        /// <summary>
        /// The forward pass of the layer normalization operation.
        /// </summary>
        /// <param name="input">The input for the layer normalization operation.</param>
        /// <returns>The output for the layer normalization operation.</returns>
        public DeepMatrix Forward(DeepMatrix input)
        {
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.operations[i] = new LayerNormalizationOperation();
                matrixArray[i] = this.operations[i].Forward(input[i]);
            });
            this.DeepOutput = new DeepMatrix(matrixArray);
            return this.DeepOutput;
        }

        /// <inheritdoc />
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