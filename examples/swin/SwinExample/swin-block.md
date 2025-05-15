public partial class PradSwinTransformerTools
{
    /// <summary>
    /// Implements a complete Swin Transformer block.
    /// </summary>
    public class SwinTransformerBlock
    {
        private readonly PradSwinTransformerTools tools;
        private readonly int dimModel;
        private readonly int numHeads;
        private readonly int windowSize;
        private readonly float mlpRatio;
        private readonly bool qkvBias;
        private readonly float dropPath;
        private readonly bool shifted;

        // Attention parameters
        private readonly PradOp qkvWeight;
        private readonly PradOp qkvBias;
        private readonly PradOp projWeight;
        private readonly PradOp projBias;
        private readonly PradOp relativePosTable;
        private readonly PradOp relativePosIndex;

        // Layer Norm parameters
        private readonly PradOp norm1Weight;
        private readonly PradOp norm1Bias;
        private readonly PradOp norm2Weight;
        private readonly PradOp norm2Bias;

        // MLP parameters
        private readonly PradOp mlpFc1Weight;
        private readonly PradOp mlpFc1Bias;
        private readonly PradOp mlpFc2Weight;
        private readonly PradOp mlpFc2Bias;

        /// <summary>
        /// Initializes a new instance of the <see cref="SwinTransformerBlock"/> class.
        /// </summary>
        public SwinTransformerBlock(
            PradSwinTransformerTools tools,
            int dimModel,
            int numHeads,
            int windowSize,
            float mlpRatio = 4.0f,
            bool qkvBias = true,
            float dropPath = 0.0f,
            bool shifted = false)
        {
            this.tools = tools;
            this.dimModel = dimModel;
            this.numHeads = numHeads;
            this.windowSize = windowSize;
            this.mlpRatio = mlpRatio;
            this.qkvBias = qkvBias;
            this.dropPath = dropPath;
            this.shifted = shifted;

            // Initialize attention parameters
            var qkvDim = 3 * dimModel;
            this.qkvWeight = InitializeWeight(new[] { dimModel, qkvDim });
            this.qkvBias = qkvBias ? InitializeWeight(new[] { qkvDim }) : null;
            this.projWeight = InitializeWeight(new[] { dimModel, dimModel });
            this.projBias = InitializeWeight(new[] { dimModel });

            // Initialize relative position parameters
            var numRelativeDistance = (2 * windowSize - 1) * (2 * windowSize - 1);
            this.relativePosTable = InitializeWeight(new[] { numRelativeDistance, numHeads });
            this.relativePosIndex = InitializeRelativePositionIndex(windowSize);

            // Initialize layer norm parameters
            this.norm1Weight = InitializeWeight(new[] { dimModel }, 1.0f);
            this.norm1Bias = InitializeWeight(new[] { dimModel });
            this.norm2Weight = InitializeWeight(new[] { dimModel }, 1.0f);
            this.norm2Bias = InitializeWeight(new[] { dimModel });

            // Initialize MLP parameters
            var mlpHiddenDim = (int)(dimModel * mlpRatio);
            this.mlpFc1Weight = InitializeWeight(new[] { dimModel, mlpHiddenDim });
            this.mlpFc1Bias = InitializeWeight(new[] { mlpHiddenDim });
            this.mlpFc2Weight = InitializeWeight(new[] { mlpHiddenDim, dimModel });
            this.mlpFc2Bias = InitializeWeight(new[] { dimModel });
        }

        /// <summary>
        /// Applies the Swin Transformer block to the input.
        /// </summary>
        /// <param name="input">Input tensor of shape [batch_size, height, width, channels].</param>
        /// <param name="shortcutSize">Size for cyclic shift in shifted window attention.</param>
        /// <returns>Transformed tensor of the same shape as input.</returns>
        public PradResult Forward(PradOp input, int? shortcutSize = null)
        {
            var inputShape = input.CurrentShape;
            var shortcut = input;

            // Layer Norm 1
            var normalized = input.LayerNorm(1e-5)
                .Then(PradOp.MulOp, this.norm1Weight.CurrentTensor)
                .Then(PradOp.AddOp, this.norm1Bias.CurrentTensor);

            // Cyclic shift if using shifted window
            var shifted = normalized;
            if (this.shifted && shortcutSize.HasValue)
            {
                shifted = this.CyclicShift(normalized, -shortcutSize.Value);
            }

            // Window partition
            var windows = this.tools.WindowPartition(shifted.PradOp, this.windowSize);

            // Reshape for attention
            var windowsReshaped = windows.Then(PradOp.ReshapeOp, new[]
            {
                windows.PradOp.CurrentShape[0],
                this.windowSize * this.windowSize,
                this.dimModel
            });

            // Window attention
            var attnOutput = this.tools.WindowAttention(
                windowsReshaped.PradOp,
                this.qkvWeight,
                this.qkvBias,
                this.projWeight,
                this.projBias,
                this.relativePosTable,
                this.relativePosIndex,
                this.numHeads,
                this.windowSize,
                this.shifted);

            // Reshape back
            var attnOutputReshaped = attnOutput.Then(PradOp.ReshapeOp, new[]
            {
                windows.PradOp.CurrentShape[0],
                this.windowSize,
                this.windowSize,
                this.dimModel
            });

            // Window reverse
            var mergedWindows = this.tools.WindowReverse(
                attnOutputReshaped.PradOp,
                this.windowSize,
                inputShape[1],
                inputShape[2]);

            // Reverse cyclic shift if needed
            if (this.shifted && shortcutSize.HasValue)
            {
                mergedWindows = this.CyclicShift(mergedWindows.PradOp, shortcutSize.Value);
            }

            // Apply drop path if needed
            var attnOutput2 = this.dropPath > 0 
                ? this.ApplyDropPath(mergedWindows.PradOp, this.dropPath) 
                : mergedWindows;

            // First residual connection
            var shortcut1 = shortcut.Add(attnOutput2.Result);

            // Layer Norm 2
            var normalized2 = shortcut1.Then(result => result.PradOp.LayerNorm(1e-5))
                .Then(PradOp.MulOp, this.norm2Weight.CurrentTensor)
                .Then(PradOp.AddOp, this.norm2Bias.CurrentTensor);

            // MLP block
            var mlpOutput = this.tools.Mlp(
                normalized2.PradOp,
                this.mlpFc1Weight,
                this.mlpFc1Bias,
                this.mlpFc2Weight,
                this.mlpFc2Bias);

            // Apply drop path to MLP output if needed
            var mlpOutput2 = this.dropPath > 0 
                ? this.ApplyDropPath(mlpOutput.PradOp, this.dropPath) 
                : mlpOutput;

            // Second residual connection
            return shortcut1.Then(PradOp.AddOp, mlpOutput2.Result);
        }

        /// <summary>
        /// Applies cyclic shift to the input tensor.
        /// </summary>
        private PradResult CyclicShift(PradOp input, int shift)
        {
            if (shift == 0) return input.NoOp();

            var shape = input.CurrentShape;
            var shifted = input.Roll(new[] { shift, shift }, new[] { 1, 2 });
            return shifted;
        }

        /// <summary>
        /// Applies drop path (stochastic depth) to the input tensor.
        /// </summary>
        private PradResult ApplyDropPath(PradOp input, float dropPath)
        {
            if (!this.training || dropPath == 0.0f) return input.NoOp();

            var shape = input.CurrentShape;
            var keepProb = 1.0f - dropPath;
            var randomTensor = new Tensor(new[] { shape[0], 1, 1, 1 });

            // Generate random mask
            var random = new Random();
            for (int i = 0; i < shape[0]; i++)
            {
                randomTensor[i, 0, 0, 0] = random.NextSingle() < keepProb ? 1.0f / keepProb : 0.0f;
            }

            return input.Mul(new PradOp(randomTensor).CurrentTensor);
        }

        private PradOp InitializeWeight(int[] shape, float fillValue = 0.0f)
        {
            var data = new float[shape.Aggregate((a, b) => a * b)];
            if (fillValue != 0.0f)
            {
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = fillValue;
                }
            }
            return new PradOp(new Tensor(shape, data));
        }

        private PradOp InitializeRelativePositionIndex(int windowSize)
        {
            var coords = new int[windowSize * windowSize, 2];
            for (int i = 0; i < windowSize; i++)
            {
                for (int j = 0; j < windowSize; j++)
                {
                    coords[i * windowSize + j, 0] = i;
                    coords[i * windowSize + j, 1] = j;
                }
            }

            var relativeCoords = new int[windowSize * windowSize, windowSize * windowSize, 2];
            for (int i = 0; i < windowSize * windowSize; i++)
            {
                for (int j = 0; j < windowSize * windowSize; j++)
                {
                    relativeCoords[i, j, 0] = coords[i, 0] - coords[j, 0];
                    relativeCoords[i, j, 1] = coords[i, 1] - coords[j, 1];
                }
            }

            var relativePosition = new int[windowSize * windowSize, windowSize * windowSize];
            for (int i = 0; i < windowSize * windowSize; i++)
            {
                for (int j = 0; j < windowSize * windowSize; j++)
                {
                    relativePosition[i, j] = (relativeCoords[i, j, 0] + windowSize - 1) 
                        * (2 * windowSize - 1) 
                        + (relativeCoords[i, j, 1] + windowSize - 1);
                }
            }

            return new PradOp(new Tensor(
                new[] { windowSize * windowSize, windowSize * windowSize },
                relativePosition.Cast<int, float>().ToArray()));
        }

        /// <summary>
        /// Gets or sets a value indicating whether the model is in training mode.
        /// </summary>
        public bool training { get; set; } = true;
    }
}