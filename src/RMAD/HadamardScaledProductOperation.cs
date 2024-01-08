//------------------------------------------------------------------------------
// <copyright file="HadamardScaledProductOperation.cs" author="ameritusweb" date="1/2/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise multiplication operation.
    /// </summary>
    public class HadamardScaledProductOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;
        private double minValue;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new HadamardScaledProductOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.input1, this.input2 }, (x, y) => new[] { this.input1, this.input2 });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateMatrixArrays[id];
            this.input1 = restored[0];
            this.input2 = restored[1];
        }

        /// <summary>
        /// Performs the forward operation for the Hadamard scaled product function.
        /// </summary>
        /// <param name="input1">The first input to the Hadamard scaled product operation.</param>
        /// <param name="input2">The second input to the Hadamard scaled product operation.</param>
        /// <param name="minValue">The minimum value to use for scaling.</param>
        /// <returns>The output of the Hadamard scaled product operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, double minValue)
        {
            this.input1 = input1;
            this.input2 = input2;
            this.minValue = minValue;
            int numRows = input1.Length;
            int numCols = input1[0].Length;

            this.Output = new Matrix(numRows, numCols);

            // Find the maximum value in both input matrices
            double maxVal = Math.Max(minValue, Math.Max(input1.ToArray().SelectMany(x => x).Max(), input2.ToArray().SelectMany(x => x).Max()));

            // Parallelize the outer loop
            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    // Calculate the Hadamard product and scale by the maximum value
                    this.Output[i][j] = (input1[i][j] * input2[i][j]) / maxVal;
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int numRows = this.input1.Length;
            int numCols = this.input1[0].Length;

            Matrix dLdInput1 = new Matrix(numRows, numCols);
            Matrix dLdInput2 = new Matrix(numRows, numCols);

            double maxVal = Math.Max(this.minValue, Math.Max(this.input1.ToArray().SelectMany(x => x).Max(), this.input2.ToArray().SelectMany(x => x).Max()));

            // Parallelize the outer loop
            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    dLdInput1[i][j] = dLdOutput[i][j] * this.input2[i][j] / maxVal;
                    dLdInput2[i][j] = dLdOutput[i][j] * this.input1[i][j] / maxVal;
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput1)
                .AddInputGradient(dLdInput2)
                .Build();
        }
    }
}
