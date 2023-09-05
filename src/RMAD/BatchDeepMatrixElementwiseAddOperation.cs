//------------------------------------------------------------------------------
// <copyright file="BatchDeepMatrixElementwiseAddOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Threading.Tasks;

    /// <summary>
    /// Batch deep matrix elementwise addition operation.
    /// </summary>
    public class BatchDeepMatrixElementwiseAddOperation : BatchOperation<DeepMatrixElementwiseAddOperation>
    {
        private BatchDeepMatrixElementwiseAddOperation(NeuralNetwork net)
            : base(net)
        {
            this.Operations = new DeepMatrixElementwiseAddOperation[net.Parameters.BatchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchDeepMatrixElementwiseAddOperation(net);
        }

        /// <summary>
        /// Performs the forward operation for the deep matrix elementwise add function.
        /// </summary>
        /// <param name="inputA">The first input to the deep matrix elementwise add operation.</param>
        /// <returns>The output of the matrix add operation.</returns>
        public DeepMatrix Forward(FourDimensionalMatrix inputA)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[inputA.Depth];
            Parallel.For(0, inputA.Depth, i =>
            {
                this.Operations[i] = new DeepMatrixElementwiseAddOperation();
                matrixArray[i] = this.Operations[i].Forward(inputA[i]);
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
