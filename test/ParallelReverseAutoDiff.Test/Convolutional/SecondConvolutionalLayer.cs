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
    /// The second convolutional layer.
    /// </summary>
    public class SecondConvolutionalLayer
    {
        private readonly ConvolutionalNeuralNetwork convolutionalNeuralNetwork;

        /// <summary>
        /// Initializes a new instance of the <see cref="SecondConvolutionalLayer"/> class.
        /// </summary>
        /// <param name="convolutionalNeuralNetwork">The neural network.</param>
        public SecondConvolutionalLayer(ConvolutionalNeuralNetwork convolutionalNeuralNetwork)
        {
            this.convolutionalNeuralNetwork = convolutionalNeuralNetwork;
        }

        /// <summary>
        /// Gets or sets the filters.
        /// </summary>
        public DeepMatrix[] Cf2 { get; set; }

        /// <summary>
        /// Gets or sets the bias for the filters.
        /// </summary>
        public Matrix Cb2 { get; set; }

        /// <summary>
        /// Gets or sets the matrix for scaling.
        /// </summary>
        public Matrix Sc2 { get; set; }

        /// <summary>
        /// Gets or sets the matrix of shifting.
        /// </summary>
        public Matrix Sh2 { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the filters with respect to the loss function.
        /// </summary>
        public DeepMatrix[] DCf2 { get; set; }

        /// <summary>
        /// Gets or sets the first moment (moving average) of the filters gradients, used in optimization algorithms like Adam.
        /// </summary>
        public DeepMatrix[] MCf2 { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the filters gradients, used in optimization algorithms like Adam.
        /// </summary>
        public DeepMatrix[] VCf2 { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the bias for the filters with respect to the loss function.
        /// </summary>
        public Matrix DCb2 { get; set; }

        /// <summary>
        /// Gets or sets the first moment (moving average) of the bias for the filters gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MCb2 { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the bias for the filters gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VCb2 { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the scaling with respect to the loss function.
        /// </summary>
        public Matrix DSc2 { get; set; }

        /// <summary>
        /// Gets or sets the first moment (moving average) of the scaling gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MSc2 { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the scaling gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VSc2 { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the shifting with respect to the loss function.
        /// </summary>
        public Matrix DSh2 { get; set; }
            
        /// <summary>
        /// Gets or sets the first moment (moving average) of the shifting gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MSh2 { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the shifting gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VSh2 { get; set; }

        /// <summary>
        /// Initialize the weights and biases and moments.
        /// </summary>
        public void Initialize()
        {
            this.Cf2 = CommonMatrixUtils.InitializeRandomMatrixWithXavierInitialization(this.convolutionalNeuralNetwork.NumFilters, this.convolutionalNeuralNetwork.InputDimensions.Depth, this.convolutionalNeuralNetwork.InputDimensions.Height, this.convolutionalNeuralNetwork.InputDimensions.Width);
            this.Cb2 = CommonMatrixUtils.InitializeZeroMatrix(this.convolutionalNeuralNetwork.NumFilters, 1);
            this.Sc2 = CommonMatrixUtils.InitializeRandomMatrixWithXavierInitialization(this.convolutionalNeuralNetwork.InputDimensions.Height, this.convolutionalNeuralNetwork.InputDimensions.Depth);
            this.Sh2 = CommonMatrixUtils.InitializeRandomMatrixWithXavierInitialization(this.convolutionalNeuralNetwork.InputDimensions.Height, this.convolutionalNeuralNetwork.InputDimensions.Depth);
            this.VCf2 = DeepMatrix.InitializeArray(this.convolutionalNeuralNetwork.NumFilters, this.convolutionalNeuralNetwork.InputDimensions.Depth, this.convolutionalNeuralNetwork.InputDimensions.Height, this.convolutionalNeuralNetwork.InputDimensions.Width);
            this.MCf2 = DeepMatrix.InitializeArray(this.convolutionalNeuralNetwork.NumFilters, this.convolutionalNeuralNetwork.InputDimensions.Depth, this.convolutionalNeuralNetwork.InputDimensions.Height, this.convolutionalNeuralNetwork.InputDimensions.Width);
            this.VCb2 = new Matrix(this.Cb2.Rows, this.Cb2.Cols);
            this.MCb2 = new Matrix(this.Cb2.Rows, this.Cb2.Cols);
            this.VSc2 = new Matrix(this.Sc2.Rows, this.Sc2.Cols);
            this.MSc2 = new Matrix(this.Sc2.Rows, this.Sc2.Cols);
            this.VSh2 = new Matrix(this.Sh2.Rows, this.Sh2.Cols);
            this.MSh2 = new Matrix(this.Sh2.Rows, this.Sh2.Cols);
        }

        /// <summary>
        /// Initialize the gradients.
        /// </summary>
        public void InitializeGradients()
        {
            this.DCf2 = DeepMatrix.InitializeArray(this.convolutionalNeuralNetwork.NumFilters, this.convolutionalNeuralNetwork.InputDimensions.Depth, this.convolutionalNeuralNetwork.InputDimensions.Height, this.convolutionalNeuralNetwork.InputDimensions.Width);
            this.DCb2 = new Matrix(this.Cb2.Rows, this.Cb2.Cols);
            this.DSc2 = new Matrix(this.Sc2.Rows, this.Sc2.Cols);
            this.DSh2 = new Matrix(this.Sh2.Rows, this.Sh2.Cols);
        }

        /// <summary>
        /// Clip the gradients.
        /// </summary>
        public void ClipGradients()
        {
            this.DCf2 = CommonMatrixUtils.ClipGradients(this.DCf2, this.convolutionalNeuralNetwork.ClipValue, null);
            this.DCb2 = CommonMatrixUtils.ClipGradients(this.DCb2, this.convolutionalNeuralNetwork.ClipValue, null);
            this.DSc2 = CommonMatrixUtils.ClipGradients(this.DSc2, this.convolutionalNeuralNetwork.ClipValue, null);
            this.DSh2 = CommonMatrixUtils.ClipGradients(this.DSh2, this.convolutionalNeuralNetwork.ClipValue, null);
        }

        /// <summary>
        /// Clear the state for the hidden layer.
        /// </summary>
        public void ClearState()
        {
            CommonMatrixUtils.ClearMatrices(this.DCf2);
            CommonMatrixUtils.ClearMatrices(new [] { this.DCb2, this.DSc2, this.DSh2 });
        }
    }
}
