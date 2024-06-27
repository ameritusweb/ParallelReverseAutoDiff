//------------------------------------------------------------------------------
// <copyright file="BatchSwigLUOperation.cs" author="ameritusweb" date="5/16/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Batch swigLU operation for a Matrix.
    /// </summary>
    public class BatchSwigLUOperation : BatchOperation<SwigLUOperation>
    {
        private readonly float beta;

        private BatchSwigLUOperation(NeuralNetwork net, float beta)
            : base(net)
        {
            this.beta = beta;
            this.Operations = new SwigLUOperation[net.Parameters.BatchSize];
        }

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchSwigLUOperation(net, net.Parameters.SwigLUBeta);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.Operations, (x, y) => this.Operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.Operations = this.IntermediateOperationArrays[id].OfType<SwigLUOperation>().ToArray();
        }

        /// <summary>
        /// The forward pass of the SwigLU operation.
        /// </summary>
        /// <param name="input">The input matrix.</param>
        /// <param name="w">Weight w.</param>
        /// <param name="v">Weight v.</param>
        /// <param name="b">Bias b.</param>
        /// <param name="c">Bias c.</param>
        /// <returns>The output matrix.</returns>
        public DeepMatrix Forward(DeepMatrix input, Matrix w, Matrix v, Matrix b, Matrix c)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.Operations[i] = new SwigLUOperation(this.beta);
                matrixArray[i] = this.Operations[i].Forward(input[i], w, v, b, c);
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
