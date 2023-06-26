//------------------------------------------------------------------------------
// <copyright file="BatchMatrixTransposeOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Threading.Tasks;

    /// <summary>
    /// A batch matrix transpose operation.
    /// </summary>
    public class BatchMatrixTransposeOperation : BatchOperation<MatrixTransposeOperation>
    {
        private MatrixTransposeOperation[] operations;

        private BatchMatrixTransposeOperation(NeuralNetwork net)
            : base(net)
        {
            this.operations = new MatrixTransposeOperation[net.Parameters.BatchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchMatrixTransposeOperation(net);
        }

        /// <summary>
        /// The forward pass of the matrix transpose operation.
        /// </summary>
        /// <param name="input">The input for the matrix transpose operation.</param>
        /// <returns>The output for the matrix transpose operation.</returns>
        public DeepMatrix Forward(DeepMatrix input)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.operations[i] = new MatrixTransposeOperation();
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
