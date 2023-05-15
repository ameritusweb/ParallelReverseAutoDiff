//------------------------------------------------------------------------------
// <copyright file="BackwardResult.cs" author="ameritusweb" date="5/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// The result of one step through a backward pass.
    /// </summary>
    public class BackwardResult
    {
        /// <summary>
        /// Gets or sets the input gradient.
        /// </summary>
        public Matrix? InputGradient { get; set; }

        /// <summary>
        /// Gets or sets the input gradient for the left input.
        /// </summary>
        public Matrix? InputGradientLeft { get; set; }

        /// <summary>
        /// Gets or sets the input gradient for the right input.
        /// </summary>
        public Matrix? InputGradientRight { get; set; }

        /// <summary>
        /// Gets or sets the deep input gradient.
        /// </summary>
        public DeepMatrix? DeepInputGradient { get; set; }

        /// <summary>
        /// Gets or sets the beta gradient.
        /// </summary>
        public Matrix? BetaGradient { get; set; }

        /// <summary>
        /// Gets or sets the gamma gradient.
        /// </summary>
        public Matrix? GammaGradient { get; set; }

        /// <summary>
        /// Gets or sets the bias gradient.
        /// </summary>
        public Matrix? BiasGradient { get; set; }

        /// <summary>
        /// Gets or sets the filters gradient.
        /// </summary>
        public DeepMatrix? FiltersGradient { get; set; }
    }
}
