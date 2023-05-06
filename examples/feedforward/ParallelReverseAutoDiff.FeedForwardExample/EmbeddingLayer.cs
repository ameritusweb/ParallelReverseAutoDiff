//------------------------------------------------------------------------------
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
        public Matrix Dbe { get; set; }

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
        public Matrix Mbe { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the bias matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix Vbe { get; set; }
    }
}
