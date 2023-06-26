//------------------------------------------------------------------------------
// <copyright file="BatchLeakyReLUOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// A Batch leaky ReLU operation.
    /// </summary>
    public class BatchLeakyReLUOperation : BatchOperation<LeakyReLUOperation>
    {
        private readonly double alpha;

        private BatchLeakyReLUOperation(NeuralNetwork net, double alpha)
            : base(net)
        {
            this.alpha = alpha;
            this.Operations = new LeakyReLUOperation[net.Parameters.BatchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchLeakyReLUOperation(net, net.Parameters.LeakyReLUAlpha);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.Operations, (key, oldValue) => this.Operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.Operations = this.IntermediateOperationArrays[id].OfType<LeakyReLUOperation>().ToArray();
        }

        /// <summary>
        /// The forward pass of the leaky ReLU operation.
        /// </summary>
        /// <param name="input">The input for the leaky ReLU operation.</param>
        /// <returns>The output for the leaky ReLU operation.</returns>
        public DeepMatrix Forward(DeepMatrix input)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.Operations[i] = new LeakyReLUOperation(this.alpha);
                matrixArray[i] = this.Operations[i].Forward(input[i]);
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
