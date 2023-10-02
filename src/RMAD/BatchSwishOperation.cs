//------------------------------------------------------------------------------
// <copyright file="BatchSwishOperation.cs" author="ameritusweb" date="6/28/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Batch swish operation for a Matrix.
    /// </summary>
    public class BatchSwishOperation : BatchOperation<SwishOperation>
    {
        private BatchSwishOperation(NeuralNetwork net)
            : base(net)
        {
            this.Operations = new SwishOperation[net.Parameters.BatchSize];
        }

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchSwishOperation(net);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.Operations, (_, _) => this.Operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.Operations = this.IntermediateOperationArrays[id].OfType<SwishOperation>().ToArray();
        }

        /// <summary>
        /// The forward pass of the Swish operation.
        /// </summary>
        /// <param name="input">The input matrix.</param>
        /// <param name="beta">Matrix beta.</param>
        /// <returns>The output matrix.</returns>
        public DeepMatrix Forward(DeepMatrix input, Matrix beta)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.Operations[i] = new SwishOperation();
                matrixArray[i] = this.Operations[i].Forward(input[i], beta);
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
                result[i] = this.Operations[i].Backward(dOutput[i]);
            });
            return result;
        }
    }
}
