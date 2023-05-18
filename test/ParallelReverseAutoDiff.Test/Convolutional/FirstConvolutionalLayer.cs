using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.Convolutional
{
    /// <summary>
    /// The first convolutional layer.
    /// </summary>
    public class FirstConvolutionalLayer
    {
        private readonly ConvolutionalNeuralNetwork convolutionalNeuralNetwork;
        private readonly FirstConvolutionalLayer? previousLayer;

        /// <summary>
        /// Initializes a new instance of the <see cref="FirstConvolutionalLayer"/> class.
        /// </summary>
        /// <param name="convolutionalNeuralNetwork">The neural network.</param>
        public FirstConvolutionalLayer(ConvolutionalNeuralNetwork convolutionalNeuralNetwork, FirstConvolutionalLayer? previousLayer)
        {
            this.convolutionalNeuralNetwork = convolutionalNeuralNetwork;
            this.previousLayer = previousLayer;
        }

        /// <summary>
        /// Gets or sets the filters.
        /// </summary>
        public DeepMatrix[] Cf1 { get; set; }

        /// <summary>
        /// Gets or sets the bias for the filters.
        /// </summary>
        public Matrix Cb1 { get; set; }

        /// <summary>
        /// Gets or sets the matrix for scaling.
        /// </summary>
        public Matrix Sc1 { get; set; }

        /// <summary>
        /// Gets or sets the matrix of shifting.
        /// </summary>
        public Matrix Sh1 { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the filters with respect to the loss function.
        /// </summary>
        public DeepMatrix[] DCf1 { get; set; }

        /// <summary>
        /// Gets or sets the first moment (moving average) of the filters gradients, used in optimization algorithms like Adam.
        /// </summary>
        public DeepMatrix[] MCf1 { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the filters gradients, used in optimization algorithms like Adam.
        /// </summary>
        public DeepMatrix[] VCf1 { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the bias for the filters with respect to the loss function.
        /// </summary>
        public Matrix DCb1 { get; set; }

        /// <summary>
        /// Gets or sets the first moment (moving average) of the bias for the filters gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MCb1 { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the bias for the filters gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VCb1 { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the scaling with respect to the loss function.
        /// </summary>
        public Matrix DSc1 { get; set; }

        /// <summary>
        /// Gets or sets the first moment (moving average) of the scaling gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MSc1 { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the scaling gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VSc1 { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the shifting with respect to the loss function.
        /// </summary>
        public Matrix DSh1 { get; set; }
            
        /// <summary>
        /// Gets or sets the first moment (moving average) of the shifting gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MSh1 { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the shifting gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VSh1 { get; set; }

        /// <summary>
        /// Initialize the weights and biases and moments.
        /// </summary>
        public void Initialize()
        {
            this.Cf1 = CommonMatrixUtils.InitializeRandomMatrixWithXavierInitialization(this.previousLayer == null ? this.convolutionalNeuralNetwork.NumFilters : this.previousLayer.Cf1.Length + this.convolutionalNeuralNetwork.NumFilters, this.previousLayer == null ? this.convolutionalNeuralNetwork.InputDimensions.Depth : this.previousLayer.Cf1.Length, this.convolutionalNeuralNetwork.FilterDimensions.Height, this.convolutionalNeuralNetwork.FilterDimensions.Width);
            this.Cb1 = CommonMatrixUtils.InitializeZeroMatrix(this.Cf1.Length, 1);
            this.Sc1 = CommonMatrixUtils.InitializeRandomMatrixWithXavierInitialization(this.previousLayer == null ? this.convolutionalNeuralNetwork.InputDimensions.Height + 3 : this.previousLayer.Sc1.Rows + 3, this.Cf1.Length);
            this.Sh1 = CommonMatrixUtils.InitializeRandomMatrixWithXavierInitialization(this.previousLayer == null ? this.convolutionalNeuralNetwork.InputDimensions.Height + 3 : this.previousLayer.Sh1.Rows + 3, this.Cf1.Length);
            this.VCf1 = DeepMatrix.InitializeArray(this.Cf1.Length, this.Cf1[0].Depth, this.Cf1[0].Rows, this.Cf1[0].Cols);
            this.MCf1 = DeepMatrix.InitializeArray(this.Cf1.Length, this.Cf1[0].Depth, this.Cf1[0].Rows, this.Cf1[0].Cols);
            this.VCb1 = new Matrix(this.Cb1.Rows, this.Cb1.Cols);
            this.MCb1 = new Matrix(this.Cb1.Rows, this.Cb1.Cols);
            this.VSc1 = new Matrix(this.Sc1.Rows, this.Sc1.Cols);
            this.MSc1 = new Matrix(this.Sc1.Rows, this.Sc1.Cols);
            this.VSh1 = new Matrix(this.Sh1.Rows, this.Sh1.Cols);
            this.MSh1 = new Matrix(this.Sh1.Rows, this.Sh1.Cols);
        }

        /// <summary>
        /// Initialize the gradients.
        /// </summary>
        public void InitializeGradients()
        {
            this.DCf1 = DeepMatrix.InitializeArray(this.Cf1.Length, this.Cf1[0].Depth, this.Cf1[0].Rows, this.Cf1[0].Cols);
            this.DCb1 = new Matrix(this.Cb1.Rows, this.Cb1.Cols);
            this.DSc1 = new Matrix(this.Sc1.Rows, this.Sc1.Cols);
            this.DSh1 = new Matrix(this.Sh1.Rows, this.Sh1.Cols);
        }

        /// <summary>
        /// Clip the gradients.
        /// </summary>
        public void ClipGradients()
        {
            this.DCf1 = CommonMatrixUtils.ClipGradients(this.DCf1, this.convolutionalNeuralNetwork.ClipValue, null);
            this.DCb1 = CommonMatrixUtils.ClipGradients(this.DCb1, this.convolutionalNeuralNetwork.ClipValue, null);
            this.DSc1 = CommonMatrixUtils.ClipGradients(this.DSc1, this.convolutionalNeuralNetwork.ClipValue, null);
            this.DSh1 = CommonMatrixUtils.ClipGradients(this.DSh1, this.convolutionalNeuralNetwork.ClipValue, null);
        }

        /// <summary>
        /// Clear the state for the hidden layer.
        /// </summary>
        public void ClearState()
        {
            CommonMatrixUtils.ClearMatrices(this.DCf1);
            CommonMatrixUtils.ClearMatrices(new [] { this.DCb1, this.DSc1, this.DSh1 });
        }
    }
}
