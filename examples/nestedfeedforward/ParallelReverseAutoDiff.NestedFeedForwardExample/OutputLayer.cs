//------------------------------------------------------------------------------
// <copyright file="OutputLayer.cs" author="ameritusweb" date="5/5/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FeedForwardExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// The output layer.
    /// </summary>
    public class OutputLayer
    {
        private readonly NestedFeedForwardNeuralNetwork feedForwardNeuralNetwork;

        /// <summary>
        /// Initializes a new instance of the <see cref="OutputLayer"/> class.
        /// </summary>
        /// <param name="feedForwardNeuralNetwork">The neural network.</param>
        public OutputLayer(NestedFeedForwardNeuralNetwork feedForwardNeuralNetwork)
        {
            this.feedForwardNeuralNetwork = feedForwardNeuralNetwork;
        }

        /// <summary>
        /// Gets or sets the weight matrix for the hidden layer.
        /// </summary>
        public Matrix V { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the hidden layer.
        /// </summary>
        public Matrix Bo { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the weight matrix with respect to the loss function.
        /// </summary>
        public Matrix DV { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the bias matrix with respect to the loss function.
        /// </summary>
        public Matrix DBo { get; set; }

        /// <summary>
        /// Gets or sets the first moment (moving average) of the weight matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MV { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the weight matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VV { get; set; }

        /// <summary>
        /// Gets or sets the first moment (moving average) of the bias matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MBo { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the bias matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VBo { get; set; }

        /// <summary>
        /// Initialize the weights, biases, and moments.
        /// </summary>
        public void Initialize()
        {
            this.V = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(this.feedForwardNeuralNetwork.OutputSize, this.feedForwardNeuralNetwork.HiddenSize);
            this.Bo = MatrixUtils.InitializeZeroMatrix(this.feedForwardNeuralNetwork.OutputSize, 1);
            this.MV = new Matrix(this.V.Rows, this.V.Cols);
            this.VV = new Matrix(this.V.Rows, this.V.Cols);
            this.MBo = new Matrix(this.Bo.Rows, this.Bo.Cols);
            this.VBo = new Matrix(this.Bo.Rows, this.Bo.Cols);
        }

        /// <summary>
        /// Initialize the gradients.
        /// </summary>
        public void InitializeGradients()
        {
            this.DV = new Matrix(this.V.Rows, this.V.Cols);
            this.DBo = new Matrix(this.Bo.Rows, this.Bo.Cols);
        }

        /// <summary>
        /// Clip the gradients.
        /// </summary>
        public void ClipGradients()
        {
            this.DV = MatrixUtils.ClipGradients(this.DV, this.feedForwardNeuralNetwork.ClipValue, null);
            this.DBo = MatrixUtils.ClipGradients(this.DBo, this.feedForwardNeuralNetwork.ClipValue, null);
        }

        /// <summary>
        /// Clear the state for the output layer.
        /// </summary>
        public void ClearState()
        {
            MatrixUtils.ClearMatrices(new[] { this.DV, this.DBo });
        }
    }
}
