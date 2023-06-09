﻿//------------------------------------------------------------------------------
// <copyright file="EmbeddingLayer.cs" author="ameritusweb" date="5/5/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FeedForwardExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// An embedding layer.
    /// </summary>
    public class EmbeddingLayer
    {
        private readonly FeedForwardNeuralNetwork feedForwardNeuralNetwork;

        /// <summary>
        /// Initializes a new instance of the <see cref="EmbeddingLayer"/> class.
        /// </summary>
        /// <param name="feedForwardNeuralNetwork">The neural network.</param>
        public EmbeddingLayer(FeedForwardNeuralNetwork feedForwardNeuralNetwork)
        {
            this.feedForwardNeuralNetwork = feedForwardNeuralNetwork;
        }

        /// <summary>
        /// Gets or sets the weight matrix for the embedding layer.
        /// </summary>
        public Matrix We { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the embedding layer.
        /// </summary>
        public Matrix Be { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the weight matrix with respect to the loss function.
        /// </summary>
        public Matrix DWe { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the bias matrix with respect to the loss function.
        /// </summary>
        public Matrix DBe { get; set; }

        /// <summary>
        /// Gets or sets the first moment (moving average) of the weight matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MWe { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the weight matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VWe { get; set; }

        /// <summary>
        /// Gets or sets the first moment (moving average) of the bias matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MBe { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the bias matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VBe { get; set; }

        /// <summary>
        /// Initialize the weights, biases, and moments for the embedding layer.
        /// </summary>
        public void Initialize()
        {
            this.We = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(this.feedForwardNeuralNetwork.HiddenSize, this.feedForwardNeuralNetwork.OriginalInputSize);
            this.Be = new Matrix(this.feedForwardNeuralNetwork.HiddenSize, 1);
            this.MWe = new Matrix(this.We.Rows, this.We.Cols);
            this.VWe = new Matrix(this.We.Rows, this.We.Cols);
            this.MBe = new Matrix(this.Be.Rows, this.Be.Cols);
            this.VBe = new Matrix(this.Be.Rows, this.Be.Cols);
        }

        /// <summary>
        /// Initialize the gradients for the embedding layer.
        /// </summary>
        public void InitializeGradients()
        {
            this.DWe = new Matrix(this.We.Rows, this.We.Cols);
            this.DBe = new Matrix(this.Be.Rows, this.Be.Cols);
        }

        /// <summary>
        /// Clip the gradients for the embedding layer.
        /// </summary>
        public void ClipGradients()
        {
            this.DWe = MatrixUtils.ClipGradients(this.DWe, this.feedForwardNeuralNetwork.ClipValue, null);
            this.DBe = MatrixUtils.ClipGradients(this.DBe, this.feedForwardNeuralNetwork.ClipValue, null);
        }

        /// <summary>
        /// Clear the state for the embedding layer.
        /// </summary>
        public void ClearState()
        {
            MatrixUtils.ClearMatrices(new[] { this.DWe, this.DBe });
        }
    }
}
