using ParallelReverseAutoDiff.PRAD;

namespace SwinExample
{
    /// <summary>
    /// Custom operations for Swin Transformer implementation using PradOp.
    /// </summary>
    public static class SwinTransformerOperations
    {
        /// <summary>
        /// Partitions the input tensor into windows.
        /// </summary>
        public class WindowPartitionOperation
        {
            private readonly int windowSize;

            public WindowPartitionOperation(int windowSize)
            {
                this.windowSize = windowSize;
            }

            public PradResult Forward(PradOp input)
            {
                var shape = input.CurrentShape;
                var B = shape[0];
                var H = shape[1];
                var W = shape[2];
                var C = shape[3];

                // Reshape input to [B, H/window_size, window_size, W/window_size, window_size, C]
                var reshaped = input.Reshape(new[] { B, H / this.windowSize, this.windowSize, W / this.windowSize, this.windowSize, C });

                // Permute to [B, H/window_size, W/window_size, window_size, window_size, C]
                var permuted = reshaped.Then(PradOp.TransposeOp, new[] { 0, 1, 3, 2, 4, 5 });

                // Final reshape to [B * num_windows, window_size * window_size, C]
                var numWindows = (H / this.windowSize) * (W / this.windowSize);
                return permuted.Then(PradOp.ReshapeOp, new[] { B * numWindows, this.windowSize * this.windowSize, C });
            }
        }

        /// <summary>
        /// Reverses the window partitioning operation.
        /// </summary>
        public class WindowReverseOperation
        {
            private readonly int windowSize;
            private readonly int height;
            private readonly int width;

            public WindowReverseOperation(int windowSize, int height, int width)
            {
                this.windowSize = windowSize;
                this.height = height;
                this.width = width;
            }

            public PradResult Forward(PradOp input)
            {
                var shape = input.CurrentShape;
                var B = shape[0] / ((this.height / this.windowSize) * (this.width / this.windowSize));
                var numH = this.height / this.windowSize;
                var numW = this.width / this.windowSize;

                // Reshape to [B, H/window_size, W/window_size, window_size, window_size, C]
                var reshaped = input.Reshape(new[] { B, numH, numW, this.windowSize, this.windowSize, shape[2] });

                // Permute to [B, H/window_size, window_size, W/window_size, window_size, C]
                var permuted = reshaped.Then(PradOp.TransposeOp, new[] { 0, 1, 3, 2, 4, 5 });

                // Final reshape to [B, H, W, C]
                return permuted.Then(PradOp.ReshapeOp, new[] { B, this.height, this.width, shape[2] });
            }
        }

        /// <summary>
        /// Implements the window attention mechanism.
        /// </summary>
        public class WindowAttentionOperation
        {
            private readonly int numHeads;
            private readonly int windowSize;
            private readonly float attentionDropout;
            private readonly bool training;

            public WindowAttentionOperation(int numHeads, int windowSize, float attentionDropout = 0.0f, bool training = true)
            {
                this.numHeads = numHeads;
                this.windowSize = windowSize;
                this.attentionDropout = attentionDropout;
                this.training = training;
            }

            public PradResult Forward(PradOp input, PradOp relativePosTable)
            {
                var shape = input.CurrentShape;
                var B = shape[0];  // B * num_windows
                var N = shape[1];  // window_size * window_size
                var C = shape[2];  // embed_dim
                var headDim = C / this.numHeads;

                // Split QKV: [B, N, 3 * C] -> 3 tensors of [B, N, num_heads, head_dim]
                var qkvReshaped = input.Reshape(new[] { B, N, 3, this.numHeads, headDim });
                var qkvPermuted = qkvReshaped.Then(PradOp.TransposeOp, new[] { 2, 0, 3, 1, 4 });

                // Split into Q, K, V
                var qSplit = qkvPermuted.Then(x => x.PradOp.Indexer("0", ":", ":", ":", ":"));
                var kSplit = qkvPermuted.Then(x => x.PradOp.Indexer("1", ":", ":", ":", ":"));
                var vSplit = qkvPermuted.Then(x => x.PradOp.Indexer("2", ":", ":", ":", ":"));

                // Compute attention scores
                var scale = new Tensor(new[] { 1 }, new float[] { 1.0f / MathF.Sqrt(headDim) });
                var qScaled = qSplit.Then(PradOp.MulOp, scale);

                // Matmul Q, K^T
                var kTransposed = kSplit.Then(PradOp.TransposeOp, new[] { 0, 1, 3, 2 });
                var attnScores = qScaled.Then(PradOp.MatMulOp, kTransposed.Result);

                // Add relative position bias
                var relativePositionIndex = GenerateRelativePositionIndex();
                var relativePositionBias = relativePosTable.Then(op => 
                    op.PradOp.Gather(relativePositionIndex)).Then(x => 
                    x.PradOp.Reshape(new[] { this.windowSize * this.windowSize, this.windowSize * this.windowSize, -1 }))
                    .Then(PradOp.TransposeOp, new[] { 2, 0, 1 });

                var attnScoresWithBias = attnScores.Then(PradOp.AddOp, relativePositionBias.Result);

                // Apply softmax
                var attnProbs = attnScoresWithBias.Then(x => x.PradOp.Softmax(-1));

                // Apply attention dropout during training
                var attnProbsDropout = this.training && this.attentionDropout > 0
                    ? attnProbs.Then(x => x.PradOp.Dropout(this.attentionDropout))
                    : attnProbs;

                // Matmul with V
                var attnOutput = attnProbsDropout.Then(PradOp.MatMulOp, vSplit.Result);

                // Reshape to original dimensions
                var outputTransposed = attnOutput.Then(PradOp.TransposeOp, new[] { 0, 2, 1, 3 });
                return outputTransposed.Then(PradOp.ReshapeOp, new[] { B, N, C });
            }

            private Tensor GenerateRelativePositionIndex()
            {
                var coords = new int[this.windowSize * this.windowSize, 2];
                for (int i = 0; i < this.windowSize; i++)
                {
                    for (int j = 0; j < this.windowSize; j++)
                    {
                        coords[i * this.windowSize + j, 0] = i;
                        coords[i * this.windowSize + j, 1] = j;
                    }
                }

                var relativeCoords = new int[this.windowSize * this.windowSize, this.windowSize * this.windowSize];
                for (int i = 0; i < this.windowSize * this.windowSize; i++)
                {
                    for (int j = 0; j < this.windowSize * this.windowSize; j++)
                    {
                        var relPosH = coords[i, 0] - coords[j, 0] + this.windowSize - 1;
                        var relPosW = coords[i, 1] - coords[j, 1] + this.windowSize - 1;
                        relativeCoords[i, j] = relPosH * (2 * this.windowSize - 1) + relPosW;
                    }
                }

                return new Tensor(new[] { this.windowSize * this.windowSize, this.windowSize * this.windowSize }, 
                    relativeCoords.Cast<int, float>().ToArray());
            }
        }

        /// <summary>
        /// Implements the GELU activation function.
        /// </summary>
        public class GELUOperation
        {
            public PradResult Forward(PradOp input)
            {
                // GELU(x) = x * Φ(x)
                // where Φ(x) is the cumulative distribution function of the standard normal distribution
                // We use the approximation: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
                
                var x = input;
                var xCubed = x.Then(PradOp.PowOp, 3.0);
                var innerTerm = xCubed.Then(PradOp.MulOp, new Tensor(x.CurrentShape, 0.044715f));
                var sumTerm = x.Then(PradOp.AddOp, innerTerm.Result);
                var constTerm = new Tensor(x.CurrentShape, MathF.Sqrt(2.0f / MathF.PI));
                var tanhTerm = sumTerm.Then(PradOp.MulOp, constTerm)
                                    .Then(PradOp.TanhOp);
                
                var onePlusTanh = tanhTerm.Then(PradOp.AddOp, new Tensor(tanhTerm.PradOp.CurrentShape, 1.0f));
                var halfX = x.Then(PradOp.MulOp, new Tensor(x.CurrentShape, 0.5f));
                
                return halfX.Then(PradOp.MulOp, onePlusTanh.Result);
            }
        }
    }
}
