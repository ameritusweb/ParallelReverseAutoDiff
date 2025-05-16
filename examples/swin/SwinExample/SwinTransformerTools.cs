using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.PRAD.Extensions;

namespace SwinExample
{
    public class SwinTransformerTools
    {
        /// <summary>
        /// Applies cyclic shift to the input feature map. Used in shifted window attention.
        /// </summary>
        /// <param name="input">Input tensor of shape [B, H, W, C]</param>
        /// <param name="shift">Amount to shift features (typically windowSize/2)</param>
        /// <returns>Shifted tensor of same shape as input</returns>
        public PradResult CyclicShift(PradOp input, int shift)
        {
            if (shift == 0)
                return input.NoOp();

            var shape = input.CurrentShape;
            var B = shape[0];
            var H = shape[1];
            var W = shape[2];
            var C = shape[3];

            // Roll along H and W dimensions
            // First roll along H
            var sliceH = input.Indexer($":", $"{H-shift}:", ":", ":");
            var remainderH = input.Indexer($":", $":{H-shift}", ":", ":");
            var rolledH = sliceH.PradOp.Concat(
                new[] { remainderH.Result },
                axis: 1);

            // Then roll along W
            var sliceW = rolledH.Then(x => x.PradOp.Indexer($":", ":", $"{W-shift}:", ":"));
            var remainderW = rolledH.Then(x => x.PradOp.Indexer($":", ":", $":{W-shift}", ":"));
            var rolledHW = sliceW.PradOp.Concat(
                new[] { remainderW.Result },
                axis: 2);

            return rolledHW;
        }

        /// <summary>
        /// Performs adaptive average pooling to convert features to target output size.
        /// </summary>
        /// <param name="input">Input tensor of shape [B, H, W, C]</param>
        /// <param name="options">Options tensor with [outputHeight, outputWidth]</param>
        /// <returns>Output tensor of shape [B, outputHeight, outputWidth, C]</returns>
        public PradResult AdaptiveAveragePool(PradOp input, PradOp options)
        {
            var shape = input.CurrentShape;
            var B = shape[0];
            var inputHeight = shape[1];
            var inputWidth = shape[2];
            var C = shape[3];

            var outputHeight = (int)options.CurrentTensor[0];
            var outputWidth = (int)options.CurrentTensor[1];

            // Calculate stride and kernel size for each dimension
            var strideH = inputHeight / outputHeight;
            var strideW = inputWidth / outputWidth;
            var kernelH = inputHeight - ((outputHeight - 1) * strideH);
            var kernelW = inputWidth - ((outputWidth - 1) * strideW);

            return input.CustomOperation(
                operation: (inputTensor) =>
                {
                    var output = new Tensor(new[] { B, outputHeight, outputWidth, C });

                    // For each output position
                    for (int b = 0; b < B; b++)
                    {
                        for (int oh = 0; oh < outputHeight; oh++)
                        {
                            for (int ow = 0; ow < outputWidth; ow++)
                            {
                                for (int c = 0; c < C; c++)
                                {
                                    // Calculate input region boundaries
                                    var startH = oh * strideH;
                                    var endH = Math.Min(startH + kernelH, inputHeight);
                                    var startW = ow * strideW;
                                    var endW = Math.Min(startW + kernelW, inputWidth);

                                    // Calculate average over the region
                                    double sum = 0;
                                    int count = 0;
                                    for (int h = startH; h < endH; h++)
                                    {
                                        for (int w = startW; w < endW; w++)
                                        {
                                            sum += inputTensor[b, h, w, c];
                                            count++;
                                        }
                                    }
                                    output[b, oh, ow, c] = sum / count;
                                }
                            }
                        }
                    }

                    return output;
                },
                reverseOperation: (inputTensor, outputTensor, gradientTensor) =>
                {
                    var inputGradient = new Tensor(inputTensor.Shape);

                    // Distribute gradient to input positions
                    for (int b = 0; b < B; b++)
                    {
                        for (int oh = 0; oh < outputHeight; oh++)
                        {
                            for (int ow = 0; ow < outputWidth; ow++)
                            {
                                // Calculate input region boundaries
                                var startH = oh * strideH;
                                var endH = Math.Min(startH + kernelH, inputHeight);
                                var startW = ow * strideW;
                                var endW = Math.Min(startW + kernelW, inputWidth);

                                // Count elements in the pooling region
                                var count = (endH - startH) * (endW - startW);
                                var scale = 1.0 / count;

                                for (int c = 0; c < C; c++)
                                {
                                    var grad = gradientTensor[b, oh, ow, c] * scale;

                                    // Distribute gradient evenly to all input positions
                                    for (int h = startH; h < endH; h++)
                                    {
                                        for (int w = startW; w < endW; w++)
                                        {
                                            inputGradient[b, h, w, c] += grad;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    return new[] { inputGradient };
                });
        }

        /// <summary>
        /// Implements MLP (FFN) block in Swin Transformer.
        /// </summary>
        /// <param name="input">Input tensor of shape [B, L, C]</param>
        /// <param name="fc1Weight">First fully connected layer weights</param>
        /// <param name="fc1Bias">First fully connected layer bias</param>
        /// <param name="fc2Weight">Second fully connected layer weights</param>
        /// <param name="fc2Bias">Second fully connected layer bias</param>
        /// <param name="options">Options tensor with [dropoutRate, training]</param>
        /// <returns>Output tensor of shape [B, L, C]</returns>
        public PradResult MLP(
            PradOp input,
            PradOp fc1Weight,
            PradOp fc1Bias,
            PradOp fc2Weight,
            PradOp fc2Bias,
            PradOp options)
        {
            var dropoutRate = (float)options.CurrentTensor[0];
            var training = options.CurrentTensor[1] > 0;

            // First fully connected layer
            var fc1 = input.MatMul(fc1Weight.CurrentTensor)
                .Then(PradOp.AddOp, fc1Bias.CurrentTensor);

            // GELU activation
            var gelu = fc1.PradOp.GELU();

            // Apply dropout if in training mode
            var geluDropout = training && dropoutRate > 0
                ? gelu.Then(x => x.PradOp.Dropout(dropoutRate))
                : gelu;

            // Second fully connected layer
            var fc2 = geluDropout.Then(PradOp.MatMulOp, fc2Weight.CurrentTensor)
                .Then(PradOp.AddOp, fc2Bias.CurrentTensor);

            // Apply dropout if in training mode
            return training && dropoutRate > 0
                ? fc2.Then(x => x.PradOp.Dropout(dropoutRate))
                : fc2;
        }

        /// <summary>
        /// Merges patches and reduces resolution by factor of 2, doubling channels.
        /// </summary>
        /// <param name="input">Input tensor of shape [B, H, W, C]</param>
        /// <param name="mergeWeight">Weight matrix for linear transformation</param>
        /// <param name="mergeBias">Bias for linear transformation</param>
        /// <param name="normWeight">Weight for layer normalization</param>
        /// <param name="normBias">Bias for layer normalization</param>
        /// <returns>Output tensor of shape [B, H/2, W/2, 2C]</returns>
        public PradResult PatchMerging(
            PradOp input,
            PradOp mergeWeight,
            PradOp mergeBias,
            PradOp normWeight,
            PradOp normBias)
        {
            var shape = input.CurrentShape;
            var B = shape[0];
            var H = shape[1];
            var W = shape[2];
            var C = shape[3];

            // Ensure dimensions are even
            if (H % 2 != 0 || W % 2 != 0)
            {
                throw new ArgumentException("Feature map dimensions must be even for patch merging");
            }

            // Get the four groups of patches using indexing
            var x0 = input.Indexer(":", "::2", "::2", ":");     // Top-left
            var x1 = input.Indexer(":", "::2", "1::2", ":");    // Top-right
            var x2 = input.Indexer(":", "1::2", "::2", ":");    // Bottom-left
            var x3 = input.Indexer(":", "1::2", "1::2", ":");   // Bottom-right

            // Concatenate along channel dimension
            var concatenated = x0.PradOp.Concat(
                new[] { x1.Result, x2.Result, x3.Result },
                axis: -1);

            // Layer normalization
            var normalized = concatenated.Then(result => result.PradOp.LayerNorm(1e-5))
                .Then(PradOp.MulOp, normWeight.CurrentTensor)
                .Then(PradOp.AddOp, normBias.CurrentTensor);

            // Linear transformation
            return normalized.Then(PradOp.MatMulOp, mergeWeight.CurrentTensor)
                .Then(PradOp.AddOp, mergeBias.CurrentTensor);
        }

        /// <summary>
        /// Reverses window partitioning to restore original feature map dimensions.
        /// </summary>
        /// <param name="input">Input tensor of shape [B*num_windows, window_size*window_size, C]</param>
        /// <param name="options">Options tensor with [windowSize, height, width]</param>
        /// <returns>Output tensor of shape [B, H, W, C]</returns>
        public PradResult WindowReverse(PradOp input, PradOp options)
        {
            var shape = input.CurrentShape;
            var windowSize = (int)options.CurrentTensor[0];
            var height = (int)options.CurrentTensor[1];
            var width = (int)options.CurrentTensor[2];
            
            var B = shape[0] / ((height / windowSize) * (width / windowSize));
            var numH = height / windowSize;
            var numW = width / windowSize;
            var C = shape[2];

            // Reshape to [B, H/window_size, W/window_size, window_size, window_size, C]
            var reshaped = input.Reshape(new[]
            {
                B,
                numH,
                numW,
                windowSize,
                windowSize,
                C
            });

            // Permute to [B, H/window_size, window_size, W/window_size, window_size, C]
            var permuted = reshaped.Then(PradOp.TransposeOp, new[] { 0, 1, 3, 2, 4, 5 });

            // Final reshape to [B, H, W, C]
            return permuted.Then(PradOp.ReshapeOp, new[] { B, height, width, C });
        }

        /// <summary>
        /// Partitions an input tensor into windows.
        /// </summary>
        /// <param name="input">Input tensor of shape [B, H, W, C]</param>
        /// <param name="options">Options tensor with [windowSize]</param>
        /// <returns>Output tensor of shape [B*num_windows, window_size*window_size, C]</returns>
        public PradResult WindowPartition(PradOp input, PradOp options)
        {
            var shape = input.CurrentShape;
            var B = shape[0];
            var H = shape[1];
            var W = shape[2];
            var C = shape[3];
            var windowSize = (int)options.CurrentTensor[0];

            // Reshape to [B, H/window_size, window_size, W/window_size, window_size, C]
            var reshaped = input.Reshape(new[] 
            { 
                B, 
                H / windowSize, 
                windowSize, 
                W / windowSize, 
                windowSize, 
                C 
            });

            // Permute to [B, H/window_size, W/window_size, window_size, window_size, C]
            var permuted = reshaped.Then(PradOp.TransposeOp, new[] { 0, 1, 3, 2, 4, 5 });

            // Final reshape to [B*num_windows, window_size*window_size, C]
            var numWindows = (H / windowSize) * (W / windowSize);
            return permuted.Then(PradOp.ReshapeOp, new[] 
            { 
                B * numWindows, 
                windowSize * windowSize, 
                C 
            });
        }

        /// <summary>
        /// Computes window attention for Swin Transformer.
        /// </summary>
        /// <param name="input">Input of shape [num_windows*B, N, C]</param>
        /// <param name="qkvWeight">Weight matrix for QKV projection</param>
        /// <param name="qkvBias">Bias for QKV projection</param>
        /// <param name="projWeight">Weight matrix for output projection</param>
        /// <param name="projBias">Bias for output projection</param>
        /// <param name="relativePosTable">Relative position bias table</param>
        /// <param name="options">Configuration tensor [numHeads, windowSize, attentionDropout, projectionDropout, training]</param>
        /// <returns>Output of same shape as input</returns>
        public PradResult WindowAttention(
            PradOp input,
            PradOp qkvWeight,
            PradOp qkvBias,
            PradOp projWeight,
            PradOp projBias,
            PradOp relativePosTable,
            PradOp options)
        {
            var shape = input.CurrentShape;
            var numWindows = shape[0];  // B * num_windows
            var numTokens = shape[1];   // N = window_size * window_size
            var embedDim = shape[2];    // C = embed_dim

            // Extract options
            var numHeads = (int)options.CurrentTensor[0];
            var windowSize = (int)options.CurrentTensor[1];
            var attentionDropout = (float)options.CurrentTensor[2];
            var projectionDropout = (float)options.CurrentTensor[3];
            var training = options.CurrentTensor[4] > 0;
            var headDim = embedDim / numHeads;

            // QKV projection [B*nW, N, C] -> [B*nW, N, 3C]
            var qkv = input.MatMul(qkvWeight.CurrentTensor)
                .Then(PradOp.AddOp, qkvBias.CurrentTensor);

            // Reshape: [B*nW, N, 3C] -> [B*nW, N, 3, num_heads, head_dim]
            var qkvReshaped = qkv.Then(PradOp.ReshapeOp,
                new[] { numWindows, numTokens, 3, numHeads, headDim });

            // Transpose: [B*nW, N, 3, num_heads, head_dim] -> [3, B*nW, num_heads, N, head_dim]
            var qkvPermuted = qkvReshaped.Then(PradOp.TransposeOp,
                new[] { 2, 0, 3, 1, 4 });

            // Split Q, K, V
            var qkvSplit = qkvPermuted.PradOp.Split(1, axis: 0);
            var query = qkvSplit[0];
            var key = qkvSplit[1];
            var value = qkvSplit[2];

            // Scale query
            var scale = new Tensor(query.CurrentShape, 1.0d / Math.Sqrt(headDim));
            var queryScaled = query.Mul(scale);

            // Compute attention scores
            var keyTransposed = key.Transpose(new[] { 0, 1, 3, 2 });
            var attnScores = queryScaled.Then(PradOp.MatMulOp, keyTransposed.Result);

            // Add relative position bias
            var relPosReshaped = relativePosTable.Reshape(new[]
            {
                1,
                numHeads,
                windowSize * windowSize,
                windowSize * windowSize
            });
            attnScores = attnScores.Then(PradOp.AddOp, relPosReshaped.Result);

            // Apply attention dropout during training
            var attnProbs = attnScores.Then(op => op.PradOp.SymmetricSoftmax(1, true, -1));
            if (training && attentionDropout > 0)
            {
                attnProbs = attnProbs.Then(op => op.PradOp.Dropout(attentionDropout));
            }

            // Attention output
            var attnOutput = attnProbs.Then(PradOp.MatMulOp, value.Result);

            // Transpose and reshape back
            var attnOutputTransposed = attnOutput.Then(PradOp.TransposeOp, new[] { 0, 2, 1, 3 });
            var attnOutputReshaped = attnOutputTransposed.Then(PradOp.ReshapeOp,
                new[] { numWindows, numTokens, numHeads * headDim });

            // Final projection
            var output = attnOutputReshaped.Then(PradOp.MatMulOp, projWeight.CurrentTensor)
                .Then(PradOp.AddOp, projBias.CurrentTensor);

            // Apply projection dropout during training
            if (training && projectionDropout > 0)
            {
                output = output.Then(op => op.PradOp.Dropout(projectionDropout));
            }

            return output;
        }

        /// <summary>
        /// Computes window attention for Swin Transformer.
        /// </summary>
        /// <param name="input">Input of shape [num_windows*B, N, C]</param>
        /// <param name="qkvWeight">Weight matrix for QKV projection</param>
        /// <param name="qkvBias">Bias for QKV projection</param>
        /// <param name="projWeight">Weight matrix for output projection</param>
        /// <param name="projBias">Bias for output projection</param>
        /// <param name="relativePosTable">Relative position bias table</param>
        /// <param name="numHeads">Number of attention heads</param>
        /// <param name="windowSize">Size of each attention window</param>
        /// <param name="training">Whether in training mode</param>
        /// <param name="attentionDropout">Dropout rate for attention (default: 0)</param>
        /// <param name="projectionDropout">Dropout rate for projection (default: 0)</param>
        /// <returns>Output of same shape as input</returns>
        public PradResult WindowAttention(
            PradOp input,
            PradOp qkvWeight,
            PradOp qkvBias,
            PradOp projWeight,
            PradOp projBias,
            PradOp relativePosTable,
            int numHeads,
            int windowSize,
            bool training = true,
            float attentionDropout = 0,
            float projectionDropout = 0)
        {
            var shape = input.CurrentShape;
            var numWindows = shape[0];  // B * num_windows
            var numTokens = shape[1];   // N = window_size * window_size
            var embedDim = shape[2];    // C = embed_dim
            var headDim = embedDim / numHeads;

            // QKV projection [B*nW, N, C] -> [B*nW, N, 3C]
            var qkv = input.MatMul(qkvWeight.CurrentTensor)
                .Then(PradOp.AddOp, qkvBias.CurrentTensor);

            // Reshape: [B*nW, N, 3C] -> [B*nW, N, 3, num_heads, head_dim]
            var qkvReshaped = qkv.Then(PradOp.ReshapeOp,
                new[] { numWindows, numTokens, 3, numHeads, headDim });

            // Transpose: [B*nW, N, 3, num_heads, head_dim] -> [3, B*nW, num_heads, N, head_dim]
            var qkvPermuted = qkvReshaped.Then(PradOp.TransposeOp,
                new[] { 2, 0, 3, 1, 4 });

            // Split Q, K, V
            var qkvSplit = qkvPermuted.PradOp.Split(1, axis: 0);
            var query = qkvSplit[0];
            var key = qkvSplit[1];
            var value = qkvSplit[2];

            // Scale query
            var scale = new Tensor(query.CurrentShape, 1.0d / Math.Sqrt(headDim));
            var queryScaled = query.Mul(scale);

            // Compute attention scores
            var keyTransposed = key.Transpose(new[] { 0, 1, 3, 2 });
            var attnScores = queryScaled.Then(PradOp.MatMulOp, keyTransposed.Result);

            // Add relative position bias
            var relPosReshaped = relativePosTable.Reshape(new[]
            {
                1,
                numHeads,
                windowSize * windowSize,
                windowSize * windowSize
            });
            attnScores = attnScores.Then(PradOp.AddOp, relPosReshaped.Result);

            // Apply attention dropout during training
            var attnProbs = attnScores.PradOp.SymmetricSoftmax(1, true, -1);
            if (training && attentionDropout > 0)
            {
                attnProbs = attnProbs.Then(op => op.PradOp.Dropout(attentionDropout));
            }

            // Attention output
            var attnOutput = attnProbs.Then(PradOp.MatMulOp, value.Result);

            // Transpose and reshape back
            var attnOutputTransposed = attnOutput.Then(PradOp.TransposeOp, new[] { 0, 2, 1, 3 });
            var attnOutputReshaped = attnOutputTransposed.Then(PradOp.ReshapeOp,
                new[] { numWindows, numTokens, numHeads * headDim });

            // Final projection
            var output = attnOutputReshaped.Then(PradOp.MatMulOp, projWeight.CurrentTensor)
                .Then(PradOp.AddOp, projBias.CurrentTensor);

            // Apply projection dropout during training
            if (training && projectionDropout > 0)
            {
                output = output.Then(op => op.PradOp.Dropout(projectionDropout));
            }

            return output;
        }
        public PradResult ClassificationHead(PradOp input, PradOp classifierWeight, PradOp classifierBias, int numClasses, bool preLogits = false)
        {
            // Global average pooling to [B, C]
            var pooled = this.AdaptiveAveragePooling(input, new[] { 1, 1 });

            // Reshape to [batch_size, channels]
            var flattened = pooled.Then(PradOp.ReshapeOp, new[] { input.CurrentShape[0], input.CurrentShape[3] });

            // Classification layer
            var logits = flattened.Then(PradOp.MatMulOp, classifierWeight.CurrentTensor)
                .Then(PradOp.AddOp, classifierBias.CurrentTensor);

            if (preLogits)
            {
                return logits;
            }

            // Apply softmax for probabilities
            return logits.PradOp.SymmetricSoftmax();
        }

        public PradResult AdaptiveAveragePooling(PradOp input, int[] outputSize)
        {
            var shape = input.CurrentShape;
            var batchSize = shape[0];
            var inputHeight = shape[1];
            var inputWidth = shape[2];
            var channels = shape[3];

            var outputHeight = outputSize[0];
            var outputWidth = outputSize[1];

            // Calculate stride and kernel size for each dimension
            var strideH = inputHeight / outputHeight;
            var strideW = inputWidth / outputWidth;
            var kernelH = inputHeight - ((outputHeight - 1) * strideH);
            var kernelW = inputWidth - ((outputWidth - 1) * strideW);

            return input.CustomOperation(
                operation: (inputTensor) =>
                {
                    var output = new Tensor(new[] { batchSize, outputHeight, outputWidth, channels });

                    // For each output position
                    for (int b = 0; b < batchSize; b++)
                    {
                        for (int oh = 0; oh < outputHeight; oh++)
                        {
                            for (int ow = 0; ow < outputWidth; ow++)
                            {
                                for (int c = 0; c < channels; c++)
                                {
                                    // Calculate input region boundaries
                                    var startH = oh * strideH;
                                    var endH = Math.Min(startH + kernelH, inputHeight);
                                    var startW = ow * strideW;
                                    var endW = Math.Min(startW + kernelW, inputWidth);

                                    // Calculate average over the region
                                    double sum = 0;
                                    int count = 0;
                                    for (int h = startH; h < endH; h++)
                                    {
                                        for (int w = startW; w < endW; w++)
                                        {
                                            sum += inputTensor[b, h, w, c];
                                            count++;
                                        }
                                    }
                                    output[b, oh, ow, c] = sum / count;
                                }
                            }
                        }
                    }

                    return output;
                },
                reverseOperation: (inputTensor, outputTensor, gradientTensor) =>
                {
                    var inputGradient = new Tensor(inputTensor.Shape);

                    // Distribute gradient to input positions
                    for (int b = 0; b < batchSize; b++)
                    {
                        for (int oh = 0; oh < outputHeight; oh++)
                        {
                            for (int ow = 0; ow < outputWidth; ow++)
                            {
                                // Calculate input region boundaries
                                var startH = oh * strideH;
                                var endH = Math.Min(startH + kernelH, inputHeight);
                                var startW = ow * strideW;
                                var endW = Math.Min(startW + kernelW, inputWidth);

                                // Count elements in the pooling region
                                var count = (endH - startH) * (endW - startW);
                                var scale = 1.0f / count;

                                for (int c = 0; c < channels; c++)
                                {
                                    var grad = gradientTensor[b, oh, ow, c] * scale;

                                    // Distribute gradient evenly to all input positions
                                    for (int h = startH; h < endH; h++)
                                    {
                                        for (int w = startW; w < endW; w++)
                                        {
                                            inputGradient[b, h, w, c] += grad;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    return new[] { inputGradient };
                });
        }

        public PradResult DropPath(PradOp input, double dropPath, int training)
        {
            if (training == 0 || dropPath == 0.0d)
            {
                return input.NoOp();
            }

            var shape = input.CurrentShape;
            var keepProb = 1.0d- dropPath;

            // Generate random mask with shape [B, 1, 1, 1]
            var maskShape = new[] { shape[0], 1, 1, 1 };
            var random = new Random();
            var maskData = new double[shape[0]];
            for (int i = 0; i < shape[0]; i++)
            {
                maskData[i] = random.NextSingle() < keepProb ? 1.0d / keepProb : 0.0d;
            }

            var mask = new Tensor(maskShape, maskData);

            // Apply mask
            return input.Mul(mask);
        }
    }
}
