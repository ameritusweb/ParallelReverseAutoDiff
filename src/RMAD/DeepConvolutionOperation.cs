//------------------------------------------------------------------------------
// <copyright file="DeepConvolutionOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// The convolution operation.
    /// </summary>
    public class DeepConvolutionOperation : DeepOperation
    {
        private DeepMatrix input;
        private DeepMatrix paddedInput;
        private DeepMatrix[] filters;
        private int padding;

        /// <summary>
        /// Initializes a new instance of the <see cref="DeepConvolutionOperation"/> class.
        /// </summary>
        /// <param name="padding">The padding to be applied.</param>
        public DeepConvolutionOperation(int padding)
        {
            this.padding = padding;
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IDeepOperation Instantiate(NeuralNetwork net)
        {
            return new DeepConvolutionOperation(net.Parameters.ConvolutionPadding);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            var data = new[] { (object)this.input, (object)this.paddedInput, (object)this.filters, (object)this.padding };
            this.IntermediateObjectArrays.AddOrUpdate(id, data, (key, oldValue) => data);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateObjectArrays[id];
            this.input = (DeepMatrix)restored[0];
            this.paddedInput = (DeepMatrix)restored[1];
            this.filters = (DeepMatrix[])restored[2];
            this.padding = (int)restored[3];
        }

        /// <summary>
        /// The forward pass of the operation.
        /// </summary>
        /// <param name="input">The input for the operation.</param>
        /// <param name="filters">The filters for the operation.</param>
        /// <param name="biases">The biases for the operation.</param>
        /// <returns>The output for the operation.</returns>
        public DeepMatrix Forward(DeepMatrix input, DeepMatrix[] filters, Matrix biases)
        {
            this.input = input;
            this.filters = filters;
            int inputHeight = input.Rows;
            int inputWidth = input.Cols;
            int inputDepth = input.Depth;

            int numFilters = filters.Length;
            int filterHeight = filters[0].Rows;
            int filterWidth = filters[0].Cols;

            // Create a new input with padding
            DeepMatrix paddedInput = new DeepMatrix(inputDepth, inputHeight + (2 * this.padding), inputWidth + (2 * this.padding));
            for (int depth = 0; depth < inputDepth; depth++)
            {
                for (int row = 0; row < inputHeight; row++)
                {
                    for (int col = 0; col < inputWidth; col++)
                    {
                        paddedInput[depth, row + this.padding, col + this.padding] = input[depth, row, col];
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
                                    sum += paddedInput[m, i + k, j + l] * filters[filter][m, k, l];
                                }
                            }
                        }

                        this.DeepOutput[filter, i, j] = sum + biases[filter][0];
                    }
                }
            }

            return this.DeepOutput;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(DeepMatrix dOutput)
        {
            int inputHeight = this.input.Rows;
            int inputWidth = this.input.Cols;
            int inputDepth = this.input.Depth;

            int numFilters = dOutput.Depth;
            int filterHeight = this.filters[0].Rows;
            int filterWidth = this.filters[0].Cols;

            // Calculate the output dimensions with padding
            int outputHeight = inputHeight + (2 * this.padding) - filterHeight + 1;
            int outputWidth = inputWidth + (2 * this.padding) - filterWidth + 1;

            // Gradient w.r.t. filters
            DeepMatrix[] dFilters = new DeepMatrix[numFilters];
            for (int filter = 0; filter < numFilters; filter++)
            {
                dFilters[filter] = new DeepMatrix(inputDepth, filterHeight, filterWidth);
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
                                    dFilters[filter][m, i, j] += this.paddedInput[m, k + i, l + j] * dOutput[filter, k, l];
                                }
                            }
                        }
                    }
                }
            }

            // Gradient w.r.t. biases
            Matrix biasGradient = new Matrix(numFilters, 1);
            for (int filter = 0; filter < numFilters; filter++)
            {
                for (int i = 0; i < outputHeight; i++)
                {
                    for (int j = 0; j < outputWidth; j++)
                    {
                        biasGradient[filter][0] += dOutput[filter, i, j];
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
                                    paddedDInput[m, i + k, j + l] += this.filters[filter][m, k, l] * dOutput[filter, i, j];
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

            return new BackwardResultBuilder()
                .AddDeepInputGradient(dInput)
                .AddFiltersGradient(dFilters)
                .AddBiasGradient(biasGradient)
                .Build();
        }
    }
}
