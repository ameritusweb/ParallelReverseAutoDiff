//------------------------------------------------------------------------------
// <copyright file="AdamOptimization.cs" author="ameritusweb" date="5/5/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FeedForwardExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// The Adam optimization for a feed forward neural network.
    /// </summary>
    public partial class FeedForwardNeuralNetwork
    {
        private readonly double beta1 = 0.9;
        private readonly double beta2 = 0.999;
        private readonly double epsilon = 1e-8;

        private void UpdateEmbeddingLayerParametersWithAdam(EmbeddingLayer embeddingLayer)
        {
            this.UpdateWeightWithAdam(embeddingLayer.We, embeddingLayer.MWe, embeddingLayer.VWe, embeddingLayer.DWe, this.beta1, this.beta2, this.epsilon);
            this.UpdateWeightWithAdam(embeddingLayer.Be, embeddingLayer.MBe, embeddingLayer.VBe, embeddingLayer.DBe, this.beta1, this.beta2, this.epsilon);
        }

        private void UpdateHiddenLayerParametersWithAdam(HiddenLayer hiddenLayer)
        {
            this.UpdateWeightWithAdam(hiddenLayer.W, hiddenLayer.MW, hiddenLayer.VW, hiddenLayer.DW, this.beta1, this.beta2, this.epsilon);
            this.UpdateWeightWithAdam(hiddenLayer.B, hiddenLayer.MB, hiddenLayer.VB, hiddenLayer.DB, this.beta1, this.beta2, this.epsilon);
        }

        private void UpdateOutputLayerParametersWithAdam(OutputLayer outputLayer)
        {
            this.UpdateWeightWithAdam(outputLayer.V, outputLayer.MV, outputLayer.VV, outputLayer.DV, this.beta1, this.beta2, this.epsilon);
            this.UpdateWeightWithAdam(outputLayer.Bo, outputLayer.MBo, outputLayer.VBo, outputLayer.DBo, this.beta1, this.beta2, this.epsilon);
        }

        private void UpdateWeightWithAdam(Matrix w, Matrix mW, Matrix vW, Matrix gradient, double beta1, double beta2, double epsilon)
        {
            // Update biased first moment estimate
            mW = MatrixUtils.MatrixAdd(MatrixUtils.ScalarMultiply(beta1, mW), MatrixUtils.ScalarMultiply(1 - beta1, gradient));

            // Update biased second raw moment estimate
            vW = MatrixUtils.MatrixAdd(MatrixUtils.ScalarMultiply(beta2, vW), MatrixUtils.ScalarMultiply(1 - beta2, MatrixUtils.HadamardProduct(gradient, gradient)));

            // Compute bias-corrected first moment estimate
            Matrix mW_hat = MatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta1, this.AdamIteration)), mW);

            // Compute bias-corrected second raw moment estimate
            Matrix vW_hat = MatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta2, this.AdamIteration)), vW);

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
