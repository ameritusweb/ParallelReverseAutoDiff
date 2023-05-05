//------------------------------------------------------------------------------
// <copyright file="CNN.cs" author="ameritusweb" date="5/5/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
/*
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RandomForestTest
{
    public class CNN
    {
        private int numFilters = 32;
        private int numLayers = 5;
        private int filterHeight = 3;
        private int filterWidth = 3;
        private int inputDepth = 12;
        private int numActions = 1024;
        private int numHiddenNodes = 1024;
        private int _poolSize;

        private double[] _gamma;
        private double[] _beta;
        private double[] _mean;
        private double[] _variance;
        private double epsilon = 1E-8;

        public CNN()
        {
            // Initialize filters
            double[,,,,] filters = new double[numLayers, numFilters, filterHeight, filterWidth, inputDepth];
            Random rand = new Random(DateTime.UtcNow.Millisecond);
            for (int layer = 0; layer < numLayers; layer++)
            {
                for (int filter = 0; filter < numFilters; filter++)
                {
                    for (int i = 0; i < filterHeight; i++)
                    {
                        for (int j = 0; j < filterWidth; j++)
                        {
                            for (int k = 0; k < inputDepth; k++)
                            {
                                filters[layer, filter, i, j, k] = rand.NextDouble() * Math.Sqrt(1.0 / (filterHeight * filterWidth * inputDepth));
                            }
                        }
                    }
                }
            }

            // Initialize biases
            double[] biases1 = new double[numFilters];
            for (int i = 0; i < numFilters; i++)
            {
                biases1[i] = rand.NextDouble() * Math.Sqrt(1.0 / numFilters);
            }

            double[] biases2 = new double[numActions];
            for (int i = 0; i < numActions; i++)
            {
                biases2[i] = rand.NextDouble() * Math.Sqrt(1.0 / numActions);
            }

            // Initialize weights
            double[,] weights = new double[numHiddenNodes, numActions];
            for (int i = 0; i < numHiddenNodes; i++)
            {
                for (int j = 0; j < numActions; j++)
                {
                    weights[i, j] = rand.NextDouble() * Math.Sqrt(1.0 / numHiddenNodes);
                }
            }

            _gamma = new double[numFilters];
            _beta = new double[numFilters];
            _mean = new double[numFilters];
            _variance = new double[numFilters];

        }

        private double[,,,] Convolve(double[,,,] inputData, double[,,,,] filters, double[] biases, double[] gammas, double[] betas, double epsilon)
        {
            int batchSize = inputData.GetLength(0);
            int numLayers = filters.GetLength(0);
            int inputHeight = inputData.GetLength(1);
            int inputWidth = inputData.GetLength(2);
            int inputDepth = inputData.GetLength(3);

            double[,,,] input = inputData;

            for (int layer = 0; layer < numLayers; layer++)
            {
                int numFilters = filters.GetLength(1);
                int filterHeight = filters.GetLength(2);
                int filterWidth = filters.GetLength(3);
                int outputHeight = inputHeight - filterHeight + 1;
                int outputWidth = inputWidth - filterWidth + 1;

                double[,,,] output = new double[batchSize, outputHeight, outputWidth, numFilters];

                for (int example = 0; example < batchSize; example++)
                {
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
                                            sum += input[example, i + k, j + l, m] * filters[layer, filter, k, l, m];
                                        }
                                    }
                                }
                                output[example, i, j, filter] = sum + biases[layer * numFilters + filter];
                            }
                        }
                    }
                }

                // Batch normalization
                for (int filter = 0; filter < numFilters; filter++)
                {
                    double mean = 0;
                    double var = 0;

                    // Calculate mean
                    for (int example = 0; example < batchSize; example++)
                    {
                        for (int i = 0; i < outputHeight; i++)
                        {
                            for (int j = 0; j < outputWidth; j++)
                            {
                                mean += output[example, i, j, filter];
                            }
                        }
                    }
                    mean /= (batchSize * outputHeight * outputWidth);

                    // Calculate variance
                    for (int example = 0; example < batchSize; example++)
                    {
                        for (int i = 0; i < outputHeight; i++)
                        {
                            for (int j = 0; j < outputWidth; j++)
                            {
                                var += Math.Pow(output[example, i, j, filter] - mean, 2);
                            }
                        }
                    }
                    var /= (batchSize * outputHeight * outputWidth);

                    // Normalize activations
                    for (int example = 0; example < batchSize; example++)
                    {
                        for (int i = 0; i < outputHeight; i++)
                        {
                            for (int j = 0; j < outputWidth; j++)
                            {
                                output[example, i, j, filter] = (output[example, i, j, filter] - mean) / Math.Sqrt(var + epsilon);
                            }
                        }
                    }

                    // Scale, and shift activations
                    for (int example = 0; example < batchSize; example++)
                    {
                        for (int i = 0; i < outputHeight; i++)
                        {
                            for (int j = 0; j < outputWidth; j++)
                            {
                                output[example, i, j, filter] = output[example, i, j, filter] * gammas[layer * numFilters + filter] + betas[layer * numFilters + filter];
                            }
                        }
                    }
                }
                // Apply activation function (e.g., ReLU)
                for (int example = 0; example < batchSize; example++)
                {
                    for (int i = 0; i < outputHeight; i++)
                    {
                        for (int j = 0; j < outputWidth; j++)
                        {
                            for (int filter = 0; filter < numFilters; filter++)
                            {
                                output[example, i, j, filter] = Math.Max(0, output[example, i, j, filter]);
                            }
                        }
                    }
                }

                // Update input, inputHeight, inputWidth, and inputDepth for the next layer
                input = output;
                inputHeight = outputHeight;
                inputWidth = outputWidth;
                inputDepth = numFilters;
            }

            return input;

        }

        //public double[,,] Predict(double[,,,] input)
        //{
        //    // Apply convolutional layers
        //    double[,,,] convOutput = Convolve(input, _filters1, _biases1, _gammas1, _betas1, _epsilon);
        //    convOutput = MaxPool(convOutput, _poolSize);
        //    convOutput = Convolve(convOutput, _filters2, _biases2, _gammas2, _betas2, _epsilon);
        //    convOutput = MaxPool(convOutput, _poolSize);

        //    // Flatten output
        //    double[,] flattened = Flatten(convOutput);

        //    // Apply fully connected layers
        //    double[,] fc1Output = Dot(flattened, _weights1) + _biases3;
        //    fc1Output = BatchNorm(fc1Output, _gammas3, _betas3, _epsilon);
        //    fc1Output = ReLU(fc1Output);
        //    double[,] fc2Output = Dot(fc1Output, _weights2) + _biases4;
        //    fc2Output = BatchNorm(fc2Output, _gammas4, _betas4, _epsilon);
        //    fc2Output = ReLU(fc2Output);

        //    // Apply final fully connected layer with softmax activation
        //    double[,] output = Dot(fc2Output, _weights3) + _biases5;
        //    output = Softmax(output);

        //    return output;
        //}


        public double[,,] PreprocessInput(int[,] chessboard)
        {
            int numRows = chessboard.GetLength(0);
            int numCols = chessboard.GetLength(1);

            // Define the channels (one for each piece type)
            int numChannels = 12;
            double[,,] input = new double[numRows, numCols, numChannels];

            // Map each piece to its corresponding channel
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    int pieceType = Math.Abs(chessboard[i, j]);
                    int channel = 0;
                    switch (pieceType)
                    {
                        case 1:
                            channel = 0; // White pawn
                            break;
                        case 2:
                            channel = 1; // White knight
                            break;
                        case 3:
                            channel = 2; // White bishop
                            break;
                        case 4:
                            channel = 3; // White rook
                            break;
                        case 5:
                            channel = 4; // White queen
                            break;
                        case 6:
                            channel = 5; // White king
                            break;
                        case 7:
                            channel = 6; // Black pawn
                            break;
                        case 8:
                            channel = 7; // Black knight
                            break;
                        case 9:
                            channel = 8; // Black bishop
                            break;
                        case 10:
                            channel = 9; // Black rook
                            break;
                        case 11:
                            channel = 10; // Black queen
                            break;
                        case 12:
                            channel = 11; // Black king
                            break;
                    }
                    if (chessboard[i, j] > 0)
                    {
                        input[i, j, channel] = 1; // White piece
                    }
                    else if (chessboard[i, j] < 0)
                    {
                        input[i, j, channel] = -1; // Black piece
                    }
                }
            }

            return input;
        }

        public double ComputeLoss(double[][] output, int[][] actions, double[] rewards, bool[][] legalMovesMask)
        {
            int batchSize = output.Length;
            double loss = 0.0;
            double[][] maskedOutput = ApplyMaskAndNormalize(output, legalMovesMask);

            for (int i = 0; i < batchSize; i++)
            {
                double expectedReward = 0.0;
                for (int j = 0; j < actions[i].Length; j++)
                {
                    expectedReward += Math.Log(maskedOutput[i][actions[i][j]]) * rewards[i];
                }
                loss -= expectedReward;
            }

            return loss / batchSize;
        }

        public double[][] ComputeGradient(double[][] output, int[][] actions, double[] rewards, bool[][] legalMovesMask)
        {
            int batchSize = output.Length;
            int numActions = output[0].Length;
            double[][] maskedOutput = ApplyMaskAndNormalize(output, legalMovesMask);
            double[][] gradient = new double[batchSize][];

            for (int i = 0; i < batchSize; i++)
            {
                gradient[i] = new double[numActions];
                for (int j = 0; j < numActions; j++)
                {
                    if (Array.IndexOf(actions[i], j) != -1)
                    {
                        gradient[i][j] = -rewards[i] / (maskedOutput[i][j] * batchSize);
                    }
                    else
                    {
                        gradient[i][j] = 0; // The gradient for non-selected actions is 0
                    }
                }
            }

            return gradient;
        }


        public double[][] ApplyMaskAndNormalize(double[][] output, bool[][] legalMovesMask)
        {
            int batchSize = output.Length;
            int numActions = output[0].Length;
            double[][] maskedOutput = new double[batchSize][];

            for (int i = 0; i < batchSize; i++)
            {
                maskedOutput[i] = new double[numActions];
                double sum = 0;

                for (int j = 0; j < numActions; j++)
                {
                    maskedOutput[i][j] = legalMovesMask[i][j] ? output[i][j] : 0;
                    sum += maskedOutput[i][j];
                }

                for (int j = 0; j < numActions; j++)
                {
                    maskedOutput[i][j] /= sum;
                }
            }

            return maskedOutput;
        }

        private double[,] BatchNorm(double[,] input, double[] gammas, double[] betas, double epsilon)
        {
            int batchSize = input.GetLength(0);
            int numFeatures = input.GetLength(1);
            double[,] output = new double[batchSize, numFeatures];

            for (int feature = 0; feature < numFeatures; feature++)
            {
                double mean = 0;
                double var = 0;

                // Calculate mean
                for (int example = 0; example < batchSize; example++)
                {
                    mean += input[example, feature];
                }
                mean /= batchSize;

                // Calculate variance
                for (int example = 0; example < batchSize; example++)
                {
                    var += Math.Pow(input[example, feature] - mean, 2);
                }
                var /= batchSize;

                // Normalize, scale, and shift activations
                for (int example = 0; example < batchSize; example++)
                {
                    output[example, feature] = (input[example, feature] - mean) / Math.Sqrt(var + epsilon);
                    output[example, feature] = output[example, feature] * gammas[feature] + betas[feature];
                }
            }

            return output;
        }

        private double[,] Softmax(double[,] input)
        {
            int batchSize = input.GetLength(0);
            int inputLength = input.GetLength(1);
            double[,] output = new double[batchSize, inputLength];

            for (int b = 0; b < batchSize; b++)
            {
                double sum = 0;

                // Calculate the exponentials and sum
                for (int i = 0; i < inputLength; i++)
                {
                    output[b, i] = Math.Exp(input[b, i]);
                    sum += output[b, i];
                }

                // Normalize the output to get probabilities
                for (int i = 0; i < inputLength; i++)
                {
                    output[b, i] /= sum;
                }
            }

            return output;
        }


        private double[,,] ReLU(double[,,] input)
        {
            int batchSize = input.GetLength(0);
            int numRows = input.GetLength(1);
            int numCols = input.GetLength(2);
            double[,,] output = new double[batchSize, numRows, numCols];

            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < numRows; i++)
                {
                    for (int j = 0; j < numCols; j++)
                    {
                        output[b, i, j] = Math.Max(0, input[b, i, j]);
                    }
                }
            }

            return output;
        }

        private double[,,,] MaxPool(double[,,,] inputData, int poolSize)
        {
            int batchSize = inputData.GetLength(0);
            int inputHeight = inputData.GetLength(1);
            int inputWidth = inputData.GetLength(2);
            int inputDepth = inputData.GetLength(3);
            int outputHeight = inputHeight / poolSize;
            int outputWidth = inputWidth / poolSize;
            double[,,,] output = new double[batchSize, outputHeight, outputWidth, inputDepth];

            for (int b = 0; b < batchSize; b++)
            {
                for (int d = 0; d < inputDepth; d++)
                {
                    for (int i = 0; i < outputHeight; i++)
                    {
                        for (int j = 0; j < outputWidth; j++)
                        {
                            double max = double.MinValue;
                            for (int k = 0; k < poolSize; k++)
                            {
                                for (int l = 0; l < poolSize; l++)
                                {
                                    max = Math.Max(max, inputData[b, i * poolSize + k, j * poolSize + l, d]);
                                }
                            }
                            output[b, i, j, d] = max;
                        }
                    }
                }
            }

            return output;
        }

        private double[,] Flatten(double[,,,] tensor)
        {
            int batchSize = tensor.GetLength(0);
            int height = tensor.GetLength(1);
            int width = tensor.GetLength(2);
            int depth = tensor.GetLength(3);
            double[,] flattened = new double[batchSize, height * width * depth];

            for (int b = 0; b < batchSize; b++)
            {
                int index = 0;
                for (int d = 0; d < depth; d++)
                {
                    for (int i = 0; i < height; i++)
                    {
                        for (int j = 0; j < width; j++)
                        {
                            flattened[b, index] = tensor[b, i, j, d];
                            index++;
                        }
                    }
                }
            }

            return flattened;
        }

        private double[,] Dot(double[,] input, double[,] weights)
        {
            int batchSize = input.GetLength(0);
            int inputCols = input.GetLength(1);
            int numRows = weights.GetLength(0);
            int numCols = weights.GetLength(1);
            double[,] output = new double[batchSize, numRows];

            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < numRows; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < inputCols; j++)
                    {
                        sum += input[b, j] * weights[i, j];
                    }
                    output[b, i] = sum;
                }
            }

            return output;
        }



    }
}
*/