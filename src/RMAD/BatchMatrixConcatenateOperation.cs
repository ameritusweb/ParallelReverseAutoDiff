//------------------------------------------------------------------------------
// <copyright file="BatchMatrixConcatenateOperation.cs" author="ameritusweb" date="6/24/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Batch matrix concatenate operation.
    /// </summary>
    public class BatchMatrixConcatenateOperation : BatchOperation<MatrixConcatenateOperation>
    {
        private BatchMatrixConcatenateOperation(NeuralNetwork net)
            : base(net)
        {
            this.Operations = new MatrixConcatenateOperation[net.Parameters.BatchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchMatrixConcatenateOperation(net);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.Operations, (key, oldValue) => this.Operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.Operations = this.IntermediateOperationArrays[id].OfType<MatrixConcatenateOperation>().ToArray();
        }

        /// <summary>
        /// Performs the forward operation for the matrix concatenate function.
        /// </summary>
        /// <param name="input">An array of matrices to concatenate. All matrices must have the same number of rows.</param>
        /// <returns>The output of the matrix concatenate operation.</returns>
        public DeepMatrix Forward(DeepMatrix[] input)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[input.Length];
            Parallel.For(0, input.Length, i =>
            {
                this.Operations[i] = new MatrixConcatenateOperation();
                matrixArray[i] = this.Operations[i].Forward(input[i]);
            });
            this.DeepOutput = new DeepMatrix(matrixArray);
            return this.DeepOutput;
        }

        /// <summary>
        /// Performs the forward operation for the matrix concatenate function.
        /// </summary>
        /// <param name="input">An array of matrices to concatenate. All matrices must have the same number of rows.</param>
        /// <returns>The output of the matrix concatenate operation.</returns>
        public DeepMatrix Forward(FourDimensionalMatrix input)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[input.Count];
            Parallel.For(0, input.Count, i =>
            {
                this.Operations[i] = new MatrixConcatenateOperation();
                matrixArray[i] = this.Operations[i].Forward(input[i]);
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
