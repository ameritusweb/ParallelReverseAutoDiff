//------------------------------------------------------------------------------
// <copyright file="AdamOptimization.cs" author="ameritusweb" date="5/5/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FeedForwardExample
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// The adam optimization for a feed forward neural network.
    /// </summary>
    public partial class FeedForwardNeuralNetwork
    {
        private void UpdateParametersWithAdam(Matrix[] dWi, Matrix[] dWf, Matrix[] dWo, Matrix[] dWc, Matrix[] dUi, Matrix[] dUf, Matrix[] dUo, Matrix[] dUc, Matrix[] dbi, Matrix[] dbf, Matrix[] dbo, Matrix[] dbc, Matrix dV, Matrix db, Matrix[] dWq, Matrix[] dWk, Matrix[] dWv, Matrix dWe, Matrix dbe)
        {
            double beta1 = 0.9;
            double beta2 = 0.999;
            double epsilon = 1e-8;

            // Use Parallel.For to parallelize the loop
            Parallel.For(0, this.NumLayers, layerIndex =>
            {
                // Update moments and apply Adam updates
                this.UpdateWeightWithAdam(this.Wi[layerIndex], this.mWi[layerIndex], this.vWi[layerIndex], dWi[layerIndex], beta1, beta2, epsilon, this.adamT);
                this.UpdateWeightWithAdam(this.Wf[layerIndex], this.mWf[layerIndex], this.vWf[layerIndex], dWf[layerIndex], beta1, beta2, epsilon, this.adamT);
                this.UpdateWeightWithAdam(this.Wc[layerIndex], this.mWc[layerIndex], this.vWc[layerIndex], dWc[layerIndex], beta1, beta2, epsilon, this.adamT);
                this.UpdateWeightWithAdam(this.Wo[layerIndex], this.mWo[layerIndex], this.vWo[layerIndex], dWo[layerIndex], beta1, beta2, epsilon, this.adamT);
            });

            this.UpdateWeightWithAdam(this.V, this.mV, this.vV, dV, beta1, beta2, epsilon, this.adamT);
            this.UpdateWeightWithAdam(this.b, this.mb, this.vb, db, beta1, beta2, epsilon, this.adamT);

            this.UpdateWeightWithAdam(this.We, this.mWe, this.vWe, dWe, beta1, beta2, epsilon, this.adamT);
            this.UpdateWeightWithAdam(this.be, this.mbe, this.vbe, dbe, beta1, beta2, epsilon, this.adamT);
        }

        private void UpdateWeightWithAdam(Matrix w, Matrix mW, Matrix vW, Matrix gradient, double beta1, double beta2, double epsilon, int t)
        {
            // Update biased first moment estimate
            mW = MatrixUtils.MatrixAdd(MatrixUtils.ScalarMultiply(beta1, mW), MatrixUtils.ScalarMultiply(1 - beta1, gradient));

            // Update biased second raw moment estimate
            vW = MatrixUtils.MatrixAdd(MatrixUtils.ScalarMultiply(beta2, vW), MatrixUtils.ScalarMultiply(1 - beta2, MatrixUtils.HadamardProduct(gradient, gradient)));

            // Compute bias-corrected first moment estimate
            Matrix mW_hat = MatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta1, t)), mW);

            // Compute bias-corrected second raw moment estimate
            Matrix vW_hat = MatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta2, t)), vW);

            // Update weights
            for (int i = 0; i < w.Length; i++)
            {
                for (int j = 0; j < w[0].Length; j++)
                {
                    double weightReductionValue = this.LearningRate * mW_hat[i][j] / (Math.Sqrt(vW_hat[i][j]) + epsilon);
                    w[i][j] -= weightReductionValue;
                }
            }
        }
    }
}
