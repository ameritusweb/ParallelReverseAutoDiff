//------------------------------------------------------------------------------
// <copyright file="HiddenLayer.cs" author="ameritusweb" date="5/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.CnnExample
{
    using ParallelReverseAutoDiff.CnnExample.Common;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// The hidden layer.
    /// </summary>
    public class HiddenLayer
    {
        private readonly ConvolutionalNeuralNetwork convolutionalNeuralNetwork;

        /// <summary>
        /// Initializes a new instance of the <see cref="HiddenLayer"/> class.
        /// </summary>
        /// <param name="convolutionalNeuralNetwork">The neural network.</param>
        public HiddenLayer(ConvolutionalNeuralNetwork convolutionalNeuralNetwork)
        {
            this.convolutionalNeuralNetwork = convolutionalNeuralNetwork;
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
        /// Gets or sets the scaling matrix for the hidden layer.
        /// </summary>
        public Matrix ScEnd2 { get; set; }

        /// <summary>
        /// Gets or sets the shifting matrix for the hidden layer.
        /// </summary>
        public Matrix ShEnd2 { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the scaling matrix with respect to the loss function.
        /// </summary>
        public Matrix DScEnd2 { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the shifting matrix with respect to the loss function.
        /// </summary>
        public Matrix DShEnd2 { get; set; }

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
        /// Gets or sets the first moment (moving average) of the scaling matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MScEnd2 { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the scaling matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VScEnd2 { get; set; }

        /// <summary>
        /// Gets or sets the first moment (moving average) of the shifting matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MShEnd2 { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the shifting matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VShEnd2 { get; set; }

        /// <summary>
        /// Initialize the weights and biases and moments.
        /// </summary>
        public void Initialize()
        {
            this.W = CommonMatrixUtils.InitializeRandomMatrixWithXavierInitialization(this.convolutionalNeuralNetwork.HiddenSize, this.convolutionalNeuralNetwork.HiddenSize);
            this.ScEnd2 = CommonMatrixUtils.InitializeRandomMatrixWithXavierInitialization(this.convolutionalNeuralNetwork.HiddenSize, 1);
            this.ShEnd2 = CommonMatrixUtils.InitializeRandomMatrixWithXavierInitialization(this.convolutionalNeuralNetwork.HiddenSize, 1);
            this.B = CommonMatrixUtils.InitializeZeroMatrix(this.convolutionalNeuralNetwork.HiddenSize, 1);
            this.H = CommonMatrixUtils.InitializeZeroMatrix(this.convolutionalNeuralNetwork.HiddenSize, 1);
            this.MW = new Matrix(this.W.Rows, this.W.Cols);
            this.VW = new Matrix(this.W.Rows, this.W.Cols);
            this.MB = new Matrix(this.B.Rows, this.B.Cols);
            this.VB = new Matrix(this.B.Rows, this.B.Cols);
            this.MShEnd2 = new Matrix(this.ShEnd2.Rows, this.ShEnd2.Cols);
            this.VShEnd2 = new Matrix(this.ShEnd2.Rows, this.ShEnd2.Cols);
            this.MScEnd2 = new Matrix(this.ScEnd2.Rows, this.ScEnd2.Cols);
            this.VScEnd2 = new Matrix(this.ScEnd2.Rows, this.ScEnd2.Cols);
        }

        /// <summary>
        /// Initialize the gradients.
        /// </summary>
        public void InitializeGradients()
        {
            this.DW = new Matrix(this.W.Rows, this.W.Cols);
            this.DB = new Matrix(this.B.Rows, this.B.Cols);
            this.DShEnd2 = new Matrix(this.ShEnd2.Rows, this.ShEnd2.Cols);
            this.DScEnd2 = new Matrix(this.ScEnd2.Rows, this.ScEnd2.Cols);
        }

        /// <summary>
        /// Clip the gradients.
        /// </summary>
        public void ClipGradients()
        {
            this.DW = CommonMatrixUtils.ClipGradients(this.DW, this.convolutionalNeuralNetwork.ClipValue, null);
            this.DB = CommonMatrixUtils.ClipGradients(this.DB, this.convolutionalNeuralNetwork.ClipValue, null);
            this.DShEnd2 = CommonMatrixUtils.ClipGradients(this.DShEnd2, this.convolutionalNeuralNetwork.ClipValue, null);
            this.DScEnd2 = CommonMatrixUtils.ClipGradients(this.DScEnd2, this.convolutionalNeuralNetwork.ClipValue, null);
        }

        /// <summary>
        /// Clear the state for the hidden layer.
        /// </summary>
        public void ClearState()
        {
            CommonMatrixUtils.ClearMatrices(new[] { this.DW, this.DB, this.H, this.ShEnd2, this.ScEnd2 });
        }
    }
}
