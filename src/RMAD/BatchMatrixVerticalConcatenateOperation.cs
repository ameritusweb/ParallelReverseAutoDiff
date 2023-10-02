//------------------------------------------------------------------------------
// <copyright file="BatchMatrixVerticalConcatenateOperation.cs" author="ameritusweb" date="5/8/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Batch matrix-vertical concatenate operation.
    /// </summary>
    public class BatchMatrixVerticalConcatenateOperation : BatchOperation<MatrixVerticalConcatenateOperation>
    {
        private BatchMatrixVerticalConcatenateOperation(NeuralNetwork net)
            : base(net)
        {
            this.Operations = new MatrixVerticalConcatenateOperation[net.Parameters.BatchSize];
        }

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchMatrixVerticalConcatenateOperation(net);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.Operations, (_, _) => this.Operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.Operations = this.IntermediateOperationArrays[id].OfType<MatrixVerticalConcatenateOperation>().ToArray();
        }

        /// <summary>
        /// The forward pass of the batch matrix-row concatenate operation.
        /// </summary>
        /// <param name="input">The input matrix.</param>
        /// <returns>The output matrix.</returns>
        public DeepMatrix Forward(DeepMatrix[] input)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[input.Length];
            Parallel.For(0, input.Length, i =>
            {
                this.Operations[i] = new MatrixVerticalConcatenateOperation();
                matrixArray[i] = this.Operations[i].Forward(input[i]);
            });
            this.DeepOutput = new DeepMatrix(matrixArray);
            return this.DeepOutput;
        }

        /// <summary>
        /// The forward pass of the batch matrix-row concatenate operation.
        /// </summary>
        /// <param name="input">The input matrix.</param>
        /// <returns>The output matrix.</returns>
        public DeepMatrix Forward(FourDimensionalMatrix input)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[input.Count];
            Parallel.For(0, input.Count, i =>
            {
                this.Operations[i] = new MatrixVerticalConcatenateOperation();
                matrixArray[i] = this.Operations[i].Forward(input[i]);
            });
            this.DeepOutput = new DeepMatrix(matrixArray);
            return this.DeepOutput;
        }

        /// <summary>
        /// Calculates the gradient of the matrix-row concatenate operation.
        /// </summary>
        /// <param name="gradOutput">The gradient of the output matrix.</param>
        /// <returns>A tuple containing the gradients.</returns>
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
