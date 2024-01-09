//------------------------------------------------------------------------------
// <copyright file="GravitationalInfluenceOnWeightsOperation.cs" author="ameritusweb" date="12/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// A gravitational influence operation.
    /// </summary>
    public class GravitationalInfluenceOnWeightsOperation : Operation
    {
        private double g;
        private Matrix inputA;  // NxM matrix
        private Matrix inputB;     // NxM matrix
        private Matrix n;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new GravitationalInfluenceOnWeightsOperation();
        }

        /// <summary>
        /// The forward pass of the gravitational influence operation.
        /// </summary>
        /// <param name="inputA">The input A.</param>
        /// <param name="inputB">The input B.</param>
        /// <param name="n">The N.</param>
        /// <param name="gravitationalConstant">The G.</param>
        /// <returns>The resultant matrix..</returns>
        public Matrix Forward(Matrix inputA, Matrix inputB, Matrix n, double gravitationalConstant)
        {
            this.inputA = inputA;
            this.inputB = inputB;
            this.n = n;
            this.g = gravitationalConstant;
            int rows = inputA.Rows;
            int cols = inputA.Cols;
            this.Output = new Matrix(rows, cols);

            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    double influenceSum = 0.0;
                    for (int k = 0; k < rows; k++)
                    {
                        for (int l = 0; l < cols; l++)
                        {
                            if (k == i && l == j)
                            {
                                continue; // Skip the element itself
                            }

                            double distance = inputB[k, l];
                            double gravitationalForce = this.ComputeGravitationalForce(inputA[i, j], inputA[k, l], distance, n[0][0]);
                            influenceSum += gravitationalForce;
                        }
                    }

                    this.Output[i, j] = influenceSum;
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int rows = this.inputA.Rows;
            int cols = this.inputA.Cols;

            Matrix dLdA = new Matrix(rows, cols);
            Matrix dLdB = new Matrix(rows, cols);
            Matrix dLdN = new Matrix(1, 1);

            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    for (int k = 0; k < rows; k++)
                    {
                        for (int l = 0; l < cols; l++)
                        {
                            if (k == i && l == j)
                            {
                                continue;
                            }

                            double distance = this.inputB[k, l];
                            double partialGradA = this.ComputePartialGradientA(this.inputA[i, j], this.inputA[k, l], distance, this.n[0][0]);
                            double partialGradB = this.ComputePartialGradientB(this.inputA[i, j], this.inputA[k, l], distance, this.n[0][0]);
                            double partialGradN = this.ComputePartialGradientN(this.inputA[i, j], this.inputA[k, l], distance, this.n[0][0]);

                            dLdA[i, j] += dOutput[i, j] * partialGradA;
                            dLdB[k, l] += dOutput[i, j] * partialGradB;
                            dLdN[0][0] += dOutput[i, j] * partialGradN;
                        }
                    }
                }
            });

            return new BackwardResultBuilder()
                .AddWeightGradient(dLdA)
                .AddWeightGradient(dLdB)
                .AddBetaGradient(dLdN)
                .Build();
        }

        private double ComputeGravitationalForce(double mass1, double mass2, double distance, double n)
        {
            if (distance == 0)
            {
                return 0; // Avoid division by zero
            }

            return this.g * (mass1 * mass2) / Math.Pow(distance, n);
        }

        private double ComputePartialGradientA(double mass1, double mass2, double distance, double n)
        {
            if (distance == 0)
            {
                return 0;
            }

            return this.g * mass2 / Math.Pow(distance, n);
        }

        private double ComputePartialGradientB(double mass1, double mass2, double distance, double n)
        {
            if (distance == 0)
            {
                return 0;
            }

            return -this.g * n * mass1 * mass2 / Math.Pow(distance, n + 1);
        }

        private double ComputePartialGradientN(double mass1, double mass2, double distance, double n)
        {
            if (distance == 0)
            {
                return 0;
            }

            return -this.g * (mass1 * mass2 / Math.Pow(distance, n)) * Math.Log(distance);
        }
    }
}
