//------------------------------------------------------------------------------
// <copyright file="HiddenLayer.cs" author="ameritusweb" date="5/5/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FeedForwardExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// The hidden layer.
    /// </summary>
    public class HiddenLayer
    {
        private readonly NestedFeedForwardNeuralNetwork feedForwardNeuralNetwork;

        /// <summary>
        /// Initializes a new instance of the <see cref="HiddenLayer"/> class.
        /// </summary>
        /// <param name="feedForwardNeuralNetwork">The neural network.</param>
        public HiddenLayer(NestedFeedForwardNeuralNetwork feedForwardNeuralNetwork)
        {
            this.feedForwardNeuralNetwork = feedForwardNeuralNetwork;
        }

        /// <summary>
        /// Gets or sets the hidden state.
        /// </summary>
        public Matrix H { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the hidden layer.
        /// </summary>
        public Matrix W { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the hidden layer.
        /// </summary>
        public Matrix B { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the weight matrix with respect to the loss function.
        /// </summary>
        public Matrix DW { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the bias matrix with respect to the loss function.
        /// </summary>
        public Matrix DB { get; set; }

        /// <summary>
        /// Gets or sets the first moment (moving average) of the weight matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MW { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the weight matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VW { get; set; }

        /// <summary>
        /// Gets or sets the first moment (moving average) of the bias matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MB { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the bias matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VB { get; set; }

        /// <summary>
        /// Initialize the weights and biases and moments.
        /// </summary>
        public void Initialize()
        {
            this.W = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(this.feedForwardNeuralNetwork.HiddenSize, this.feedForwardNeuralNetwork.HiddenSize);
            this.B = MatrixUtils.InitializeZeroMatrix(this.feedForwardNeuralNetwork.HiddenSize, 1);
            this.H = MatrixUtils.InitializeZeroMatrix(this.feedForwardNeuralNetwork.HiddenSize, 1);
            this.MW = new Matrix(this.W.Rows, this.W.Cols);
            this.VW = new Matrix(this.W.Rows, this.W.Cols);
            this.MB = new Matrix(this.B.Rows, this.B.Cols);
            this.VB = new Matrix(this.B.Rows, this.B.Cols);
        }

        /// <summary>
        /// Initialize the gradients.
        /// </summary>
        public void InitializeGradients()
        {
            this.DW = new Matrix(this.W.Rows, this.W.Cols);
            this.DB = new Matrix(this.B.Rows, this.B.Cols);
        }

        /// <summary>
        /// Clip the gradients.
        /// </summary>
        public void ClipGradients()
        {
            this.DW = MatrixUtils.ClipGradients(this.DW, this.feedForwardNeuralNetwork.ClipValue, null);
            this.DB = MatrixUtils.ClipGradients(this.DB, this.feedForwardNeuralNetwork.ClipValue, null);
        }

        /// <summary>
        /// Clear the state for the hidden layer.
        /// </summary>
        public void ClearState()
        {
            MatrixUtils.ClearMatrices(new[] { this.DW, this.DB, this.H });
        }
    }
}
