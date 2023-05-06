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
    }
}
