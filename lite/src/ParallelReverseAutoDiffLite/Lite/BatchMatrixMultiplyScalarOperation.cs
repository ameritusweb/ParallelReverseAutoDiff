//------------------------------------------------------------------------------
// <copyright file="BatchMatrixMultiplyScalarOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// A Batch matrix multiply scalar operation.
    /// </summary>
    public class BatchMatrixMultiplyScalarOperation : BatchOperation<MatrixMultiplyScalarOperation>
    {
        private BatchMatrixMultiplyScalarOperation(NeuralNetwork net)
            : base(net)
        {
            this.Operations = new MatrixMultiplyScalarOperation[net.Parameters.BatchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchMatrixMultiplyScalarOperation(net);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.Operations, (x, y) => this.Operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.Operations = this.IntermediateOperationArrays[id].OfType<MatrixMultiplyScalarOperation>().ToArray();
        }

        /// <summary>
        /// Performs the forward operation for the matrix multiply scalar function.
        /// </summary>
        /// <param name="input">The first input to the matrix multiply scalar operation.</param>
        /// <param name="scalar">The second input to the matrix multiply scalar operation.</param>
        /// <returns>The output of the matrix multiply scalar operation.</returns>
        public DeepMatrix Forward(DeepMatrix input, float scalar)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.Operations[i] = new MatrixMultiplyScalarOperation();
                matrixArray[i] = this.Operations[i].Forward(input[i], scalar);
            });
            this.DeepOutput = new DeepMatrix(matrixArray);
            return this.DeepOutput;
        }

        /// <summary>
        /// Performs the forward operation for the matrix multiply scalar function.
        /// </summary>
        /// <param name="input">The first input to the matrix multiply scalar operation.</param>
        /// <param name="scalar">The second input to the matrix multiply scalar operation.</param>
        /// <returns>The output of the matrix multiply scalar operation.</returns>
        public DeepMatrix Forward(DeepMatrix input, Matrix scalar)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.Operations[i] = new MatrixMultiplyScalarOperation();
                matrixArray[i] = this.Operations[i].Forward(input[i], scalar[i][0]);
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
                result[i] = this.Operations[i].Backward(dLdOutput[i]);
            });
            return result;
        }
    }
}
