public partial class PradSwinTransformerTools
{
    /// <summary>
    /// Implements efficient batch computation for window attention operations.
    /// </summary>
    public class BatchWindowAttention
    {
        private readonly int windowSize;
        private readonly int numHeads;
        private readonly int headDim;
        private readonly float attentionDropout;
        private readonly float projectionDropout;
        private readonly bool shifted;

        private readonly PradOp qkvWeight;
        private readonly PradOp qkvBias;
        private readonly PradOp projWeight;
        private readonly PradOp projBias;
        private readonly RelativePositionBias relativePosModule;

        /// <summary>
        /// Initializes a new instance of the <see cref="BatchWindowAttention"/> class.
        /// </summary>
        public BatchWindowAttention(
            int windowSize,
            int numHeads,
            int embedDim,
            float attentionDropout = 0.0f,
            float projectionDropout = 0.0f,
            bool shifted = false)
        {
            this.windowSize = windowSize;
            this.numHeads = numHeads;
            this.headDim = embedDim / numHeads;
            this.attentionDropout = attentionDropout;
            this.projectionDropout = projectionDropout;
            this.shifted = shifted;

            // Initialize weights
            var qkvDim = 3 * embedDim;
            this.qkvWeight = new PradOp(new Tensor(new[] { embedDim, qkvDim }));
            this.qkvBias = new PradOp(new Tensor(new[] { qkvDim }));
            this.projWeight = new PradOp(new Tensor(new[] { embedDim, embedDim }));
            this.projBias = new PradOp(new Tensor(new[] { embedDim }));

            // Initialize relative position bias
            this.relativePosModule = new RelativePositionBias(windowSize, numHeads);
        }

        /// <summary>
        /// Efficiently processes batched windows through attention.
        /// </summary>
        public PradResult Forward(PradOp input, WindowPartitionInfo partitionInfo)
        {
            var batchedWindows = PartitionIntoWindows(input, partitionInfo);
            var attnOutput = ComputeBatchedAttention(batchedWindows);
            return MergeWindows(attnOutput, partitionInfo);
        }

        private PradResult PartitionIntoWindows(PradOp input, WindowPartitionInfo partitionInfo)
        {
            var shape = input.CurrentShape;
            var B = shape[0];
            var H = shape[1];
            var W = shape[2];
            var C = shape[3];

            // Apply cyclic shift if needed
            var shiftedInput = input;
            if (this.shifted)
            {
                var shift = windowSize / 2;
                shiftedInput = CyclicShiftBatch(input, shift);
            }

            // Efficient window partitioning using reshape and transpose
            return shiftedInput.Then(x => x.PradOp.CustomOperation(
                operation: inputTensor =>
                {
                    // Reshape to [B, num_windows_h, window_size, num_windows_w, window_size, C]
                    var windowsH = H / windowSize;
                    var windowsW = W / windowSize;
                    
                    var reshaped = inputTensor.Reshape(new[]
                    {
                        B,
                        windowsH,
                        windowSize,
                        windowsW,
                        windowSize,
                        C
                    });

                    // Transpose and reshape to [B*num_windows, window_size*window_size, C]
                    var permuted = reshaped.Transpose(new[] { 0, 1, 3, 2, 4, 5 });
                    return permuted.Reshape(new[]
                    {
                        B * windowsH * windowsW,
                        windowSize * windowSize,
                        C
                    });
                },
                reverseOperation: (input, output, gradient) =>
                {
                    var windowsH = H / windowSize;
                    var windowsW = W / windowSize;
                    
                    // Reshape gradient back to original window structure
                    var gradReshaped = gradient.Reshape(new[]
                    {
                        B,
                        windowsH,
                        windowsW,
                        windowSize,
                        windowSize,
                        C
                    });

                    // Inverse transpose
                    var gradPermuted = gradReshaped.Transpose(new[] { 0, 1, 3, 2, 4, 5 });
                    var gradFinal = gradPermuted.Reshape(new[] { B, H, W, C });

                    return new[] { gradFinal };
                }));
        }

        private PradResult ComputeBatchedAttention(PradResult batchedWindows)
        {
            var shape = batchedWindows.PradOp.CurrentShape;
            var numWindows = shape[0];
            var numTokens = shape[1];

            // QKV projection with batched matrix multiplication
            var qkv = batchedWindows.Then(PradOp.MatMulOp, this.qkvWeight.CurrentTensor)
                .Then(PradOp.AddOp, this.qkvBias.CurrentTensor);

            // Reshape and transpose for multi-head attention
            var qkvReshaped = qkv.Then(PradOp.ReshapeOp,
                new[] { numWindows, numTokens, 3, this.numHeads, this.headDim });
            
            var qkvPermuted = qkvReshaped.Then(PradOp.TransposeOp,
                new[] { 2, 0, 3, 1, 4 }); // [3, num_windows, num_heads, tokens, head_dim]

            // Split Q, K, V
            var qkvSplit = qkvPermuted.Then(op => op.PradOp.Split(1, axis: 0));
            var query = qkvSplit[0];
            var key = qkvSplit[1];
            var value = qkvSplit[2];

            // Scaled dot-product attention
            var scale = new Tensor(query.PradOp.CurrentShape, 1.0f / Math.Sqrt(this.headDim));
            var attnScores = query.PradOp.MatMul(
                key.PradOp.Transpose(new[] { 0, 1, 3, 2 }).Result)
                .Then(PradOp.MulOp, scale);

            // Add relative position bias
            var relativePosBias = this.relativePosModule.GetRelativePositionBias()
                .Then(op => op.PradOp.Reshape(new[] 
                { 
                    1, 
                    this.numHeads, 
                    this.windowSize * this.windowSize, 
                    this.windowSize * this.windowSize 
                }));
            
            attnScores = attnScores.Then(PradOp.AddOp, relativePosBias.Result);

            // Apply attention dropout during training
            if (this.attentionDropout > 0 && this.training)
            {
                attnScores = attnScores.Then(op => op.PradOp.Dropout(this.attentionDropout));
            }

            // Attention weights
            var attnWeights = attnScores.Then(op => op.PradOp.Softmax(-1));

            // Apply attention to values
            var attnOutput = attnWeights.PradOp.MatMul(value.Result);

            // Reshape and transpose back
            var attnOutputTransposed = attnOutput.Then(PradOp.TransposeOp,
                new[] { 0, 2, 1, 3 });
            
            var attnOutputReshaped = attnOutputTransposed.Then(PradOp.ReshapeOp,
                new[] { numWindows, numTokens, this.numHeads * this.headDim });

            // Final projection with dropout
            var output = attnOutputReshaped.Then(PradOp.MatMulOp, this.projWeight.CurrentTensor)
                .Then(PradOp.AddOp, this.projBias.CurrentTensor);

            if (this.projectionDropout > 0 && this.training)
            {
                output = output.Then(op => op.PradOp.Dropout(this.projectionDropout));
            }

            return output;
        }

        private PradResult MergeWindows(PradResult attnOutput, WindowPartitionInfo partitionInfo)
        {
            var B = partitionInfo.BatchSize;
            var H = partitionInfo.Height;
            var W = partitionInfo.Width;
            var C = partitionInfo.Channels;
            var windowsH = H / this.windowSize;
            var windowsW = W / this.windowSize;

            return attnOutput.Then(x => x.PradOp.CustomOperation(
                operation: inputTensor =>
                {
                    // Reshape from [B*num_windows, window_size*window_size, C] to
                    // [B, num_windows_h, num_windows_w, window_size, window_size, C]
                    var reshaped = inputTensor.Reshape(new[]
                    {
                        B,
                        windowsH,
                        windowsW,
                        this.windowSize,
                        this.windowSize,
                        C
                    });

                    // Permute back to original order
                    var permuted = reshaped.Transpose(new[] { 0, 1, 3, 2, 4, 5 });
                    
                    // Final reshape to [B, H, W, C]
                    return permuted.Reshape(new[] { B, H, W, C });
                },
                reverseOperation: (input, output, gradient) =>
                {
                    // Reshape gradient for window structure
                    var gradReshaped = gradient.Reshape(new[]
                    {
                        B,
                        windowsH,
                        this.windowSize,
                        windowsW,
                        this.windowSize,
                        C
                    });

                    // Inverse permute
                    var gradPermuted = gradReshaped.Transpose(new[] { 0, 1, 3, 2, 4, 5 });
                    
                    // Reshape to window format
                    var gradFinal = gradPermuted.Reshape(new[]
                    {
                        B * windowsH * windowsW,
                        this.windowSize * this.windowSize,
                        C
                    });

                    return new[] { gradFinal };
                }));
        }

        private PradResult CyclicShiftBatch(PradOp input, int shift)
        {
            return input.CustomOperation(
                operation: inputTensor =>
                {
                    var shape = inputTensor.Shape;
                    var output = new Tensor(shape);

                    // Efficient batch cyclic shift
                    for (int b = 0; b < shape[0]; b++)
                    {
                        for (int h = 0; h < shape[1]; h++)
                        {
                            for (int w = 0; w < shape[2]; w++)
                            {
                                for (int c = 0; c < shape[3]; c++)
                                {
                                    var newH = (h + shift) % shape[1];
                                    var newW = (w + shift) % shape[2];
                                    output[b, newH, newW, c] = inputTensor[b, h, w, c];
                                }
                            }
                        }
                    }
                    return output;
                },
                reverseOperation: (input, output, gradient) =>
                {
                    var shape = input.Shape;
                    var gradInput = new Tensor(shape);

                    // Inverse cyclic shift for gradient
                    for (int b = 0; b < shape[0]; b++)
                    {
                        for (int h = 0; h < shape[1]; h++)
                        {
                            for (int w = 0; w < shape[2]; w++)
                            {
                                for (int c = 0; c < shape[3]; c++)
                                {
                                    var origH = (h - shift + shape[1]) % shape[1];
                                    var origW = (w - shift + shape[2]) % shape[2];
                                    gradInput[b, h, w, c] = gradient[b, origH, origW, c];
                                }
                            }
                        }
                    }
                    return new[] { gradInput };
                });
        }

        /// <summary>
        /// Gets or sets a value indicating whether the model is in training mode.
        /// </summary>
        public bool training { get; set; } = true;
    }

    /// <summary>
    /// Stores information about window partitioning.
    /// </summary>
    public class WindowPartitionInfo
    {
        public int BatchSize { get; set; }
        public int Height { get; set; }
        public int Width { get; set; }
        public int Channels { get; set; }
    }
}