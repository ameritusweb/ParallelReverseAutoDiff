﻿//------------------------------------------------------------------------------
// <copyright file="NeuralNetworkParameters.cs" author="ameritusweb" date="5/7/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

    /// <summary>
    /// The parameters for a neural network.
    /// </summary>
    public class NeuralNetworkParameters
    {
        /// <summary>
        /// Gets or sets the batch size.
        /// </summary>
        public int BatchSize { get; set; } = 8;

        /// <summary>
        /// Gets or sets the dropout rate for the apply dropout operation.
        /// </summary>
        public float DropoutRate { get; set; } = 0.01f;

        /// <summary>
        /// Gets or sets the discount factor.
        /// </summary>
        public float DiscountFactor { get; set; } = 0.99f;

        /// <summary>
        /// Gets or sets the alpha value for the LeakyReLU operation.
        /// </summary>
        public float LeakyReLUAlpha { get; set; } = 0.01f;

        /// <summary>
        /// Gets or sets the learning rate.
        /// </summary>
        public float LearningRate { get; set; } = 0.001f;

        /// <summary>
        /// Gets or sets the noise ratio for the AddGaussianNoise operation.
        /// </summary>
        public float NoiseRatio { get; set; } = 0.01f;

        /// <summary>
        /// Gets or sets the pool size for the max pool operation.
        /// </summary>
        public int PoolSize { get; set; } = 2;

        /// <summary>
        /// Gets or sets the convolution padding for the convolution operation.
        /// </summary>
        public int ConvolutionPadding { get; set; } = 2;

        /// <summary>
        /// Gets or sets the beta value for the SwigLU operation.
        /// </summary>
        public float SwigLUBeta { get; set; } = 1f;

        /// <summary>
        /// Gets or sets the Adam iteration.
        /// </summary>
        public float AdamIteration { get; set; } = 1f;

        /// <summary>
        /// Gets or sets the Adam beta 1.
        /// </summary>
        public float AdamBeta1 { get; set; } = 0.9f;

        /// <summary>
        /// Gets or sets the Adam beta 2.
        /// </summary>
        public float AdamBeta2 { get; set; } = 0.999f;

        /// <summary>
        /// Gets or sets the Adam epsilon value.
        /// </summary>
        public float AdamEpsilon { get; set; } = 1E-8f;

        /// <summary>
        /// Gets or sets the clip value.
        /// </summary>
        public float ClipValue { get; set; } = 4;

        /// <summary>
        /// Gets or sets the minimum clip value.
        /// </summary>
        public float MinimumClipValue { get; set; } = 1E-8F;

        /// <summary>
        /// Gets or sets the number of time steps.
        /// </summary>
        public int NumTimeSteps { get; set; }

        /// <summary>
        /// Gets or sets the input sequence.
        /// </summary>
        public DeepMatrix InputSequence { get; set; }

        /// <summary>
        /// Gets or sets the deep input sequence.
        /// </summary>
        public FourDimensionalMatrix DeepInputSequence { get; set; }

        /// <summary>
        /// Gets or sets the rewards for policy gradient optimization.
        /// </summary>
        public List<float> Rewards { get; set; }

        /// <summary>
        /// Gets or sets the chosen actions for policy gradient optimization.
        /// </summary>
        public List<Matrix> ChosenActions { get; set; }
    }
}