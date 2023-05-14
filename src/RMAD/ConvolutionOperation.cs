//------------------------------------------------------------------------------
// <copyright file="ConvolutionOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// The convolution operation.
    /// </summary>
    public class ConvolutionOperation : DeepOperation, IDeepOperation
    {
        private DeepMatrix input;
        private DeepMatrix paddedInput;
        private DeepMatrix filters;
        private int padding;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IDeepOperation Instantiate(NeuralNetwork net)
        {
            return new ConvolutionOperation();
        }

        /// <summary>
        /// The forward pass of the operation.
        /// </summary>
        /// <param name="input">The input for the operation.</param>
        /// <param name="filters">The filters for the operation.</param>
        /// <param name="biases">The biases for the operation.</param>
        /// <param name="padding">The padding for the operation.</param>
        /// <returns>The output for the operation.</returns>
        public DeepMatrix Forward(DeepMatrix input, DeepMatrix filters, double[] biases, int padding)
        {
            this.input = input;
            this.filters = filters;
            int inputHeight = input.Rows;
            int inputWidth = input.Cols;
            int inputDepth = input.Depth;

            int numFilters = filters.Depth;
            int filterHeight = filters.Rows;
            int filterWidth = filters.Cols;

            // Create a new input with padding
            DeepMatrix paddedInput = new DeepMatrix(inputDepth, inputHeight + (2 * padding), inputWidth + (2 * padding));
            for (int depth = 0; depth < inputDepth; depth++)
            {
                for (int row = 0; row < inputHeight; row++)
                {
                    for (int col = 0; col < inputWidth; col++)
                    {
                        paddedInput[depth, row + padding, col + padding] = input[depth, row, col];
                    }
                }
            }

            this.paddedInput = paddedInput;

            int paddedInputHeight = paddedInput.Rows;
            int paddedInputWidth = paddedInput.Cols;

            int outputHeight = paddedInputHeight - filterHeight + 1;
            int outputWidth = paddedInputWidth - filterWidth + 1;

            this.DeepOutput = new DeepMatrix(numFilters, outputHeight, outputWidth);

            for (int filter = 0; filter < numFilters; filter++)
            {
                for (int i = 0; i < outputHeight; i++)
                {
                    for (int j = 0; j < outputWidth; j++)
                    {
                        double sum = 0;
                        for (int k = 0; k < filterHeight; k++)
                        {
                            for (int l = 0; l < filterWidth; l++)
                            {
                                for (int m = 0; m < inputDepth; m++)
                                {
                                    sum += paddedInput[m, i + k, j + l] * filters[filter, k, l];
                                }
                            }
                        }

                        this.DeepOutput[filter, i, j] = sum + biases[filter];
                    }
                }
            }

            return this.DeepOutput;
        }

        /// <inheritdoc />
        public override (DeepMatrix?, DeepMatrix?) Backward(DeepMatrix dOutput)
        {
            int inputHeight = this.input.Rows;
            int inputWidth = this.input.Cols;
            int inputDepth = this.input.Depth;

            int numFilters = dOutput.Depth;
            int filterHeight = this.filters.Rows;
            int filterWidth = this.filters.Cols;

            // Calculate the output dimensions with padding
            int outputHeight = inputHeight + (2 * this.padding) - filterHeight + 1;
            int outputWidth = inputWidth + (2 * this.padding) - filterWidth + 1;

            // Gradient w.r.t. filters
            DeepMatrix dFilters = new DeepMatrix(numFilters, filterHeight, filterWidth);
            for (int filter = 0; filter < numFilters; filter++)
            {
                for (int i = 0; i < filterHeight; i++)
                {
                    for (int j = 0; j < filterWidth; j++)
                    {
                        for (int k = 0; k < outputHeight; k++)
                        {
                            for (int l = 0; l < outputWidth; l++)
                            {
                                for (int m = 0; m < inputDepth; m++)
                                {
                                    // Use the padded input
                                    dFilters[filter, i, j] += this.paddedInput[m, k + i, l + j] * dOutput[filter, k, l];
                                }
                            }
                        }
                    }
                }
            }

            // Gradient w.r.t. biases
            this.BiasGradient = new double[numFilters];
            for (int filter = 0; filter < numFilters; filter++)
            {
                for (int i = 0; i < outputHeight; i++)
                {
                    for (int j = 0; j < outputWidth; j++)
                    {
                        this.BiasGradient[filter] += dOutput[filter, i, j];
                    }
                }
            }

            // Gradient w.r.t. input (initially with padding)
            DeepMatrix paddedDInput = new DeepMatrix(inputDepth, inputHeight + (2 * this.padding), inputWidth + (2 * this.padding));
            for (int filter = 0; filter < numFilters; filter++)
            {
                for (int i = 0; i < outputHeight; i++)
                {
                    for (int j = 0; j < outputWidth; j++)
                    {
                        for (int k = 0; k < filterHeight; k++)
                        {
                            for (int l = 0; l < filterWidth; l++)
                            {
                                for (int m = 0; m < inputDepth; m++)
                                {
                                    paddedDInput[m, i + k, j + l] += this.filters[filter, k, l] * dOutput[filter, i, j];
                                }
                            }
                        }
                    }
                }
            }

            // Remove padding from the gradient w.r.t. input
            DeepMatrix dInput = new DeepMatrix(inputDepth, inputHeight, inputWidth);
            for (int depth = 0; depth < inputDepth; depth++)
            {
                for (int row = 0; row < inputHeight; row++)
                {
                    for (int col = 0; col < inputWidth; col++)
                    {
                        dInput[depth, row, col] = paddedDInput[depth, row + this.padding, col + this.padding];
                    }
                }
            }

            return (dInput, dFilters);
        }
    }
}
