﻿//------------------------------------------------------------------------------
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
    public class BatchLayerNormalizationOperation : BatchOperation<LayerNormalizationOperation>
    {
        private BatchLayerNormalizationOperation(NeuralNetwork net)
            : base(net)
        {
            this.Operations = new LayerNormalizationOperation[net.Parameters.BatchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchLayerNormalizationOperation(net);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.Operations, (x, y) => this.Operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.Operations = this.IntermediateOperationArrays[id].OfType<LayerNormalizationOperation>().ToArray();
        }

        /// <summary>
        /// The forward pass of the layer normalization operation.
        /// </summary>
        /// <param name="input">The input for the layer normalization operation.</param>
        /// <returns>The output for the layer normalization operation.</returns>
        public DeepMatrix Forward(DeepMatrix input)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.Operations[i] = new LayerNormalizationOperation();
                matrixArray[i] = this.Operations[i].Forward(input[i]);
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
                result[i] = this.Operations[i].Backward(gradOutput[i]);
            });
            return result;
        }
    }
}