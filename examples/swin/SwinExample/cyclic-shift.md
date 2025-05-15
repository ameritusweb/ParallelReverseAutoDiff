public partial class PradSwinTransformerTools
{
    /// <summary>
    /// Implements a stage in the Swin Transformer architecture.
    /// </summary>
    public class SwinTransformerStage
    {
        private readonly PradSwinTransformerTools tools;
        private readonly int depth;
        private readonly int numHeads;
        private readonly int windowSize;
        private readonly int embedDim;
        private readonly float mlpRatio;
        private readonly bool downsample;
        
        private readonly PradOp patchMergeWeight;
        private readonly PradOp patchMergeBias;
        private readonly PradOp patchMergeNormWeight;
        private readonly PradOp patchMergeNormBias;
        private readonly List<SwinTransformerBlock> blocks;

        public SwinTransformerStage(
            PradSwinTransformerTools tools,
            int depth,
            int numHeads,
            int windowSize,
            int embedDim,
            float mlpRatio = 4.0f,
            bool downsample = true)
        {
            this.tools = tools;
            this.depth = depth;
            this.numHeads = numHeads;
            this.windowSize = windowSize;
            this.embedDim = embedDim;
            this.mlpRatio = mlpRatio;
            this.downsample = downsample;

            // Initialize patch merging layers if downsampling
            if (downsample)
            {
                this.patchMergeWeight = new PradOp(new Tensor(
                    new[] { 4 * embedDim, 2 * embedDim }));
                this.patchMergeBias = new PradOp(new Tensor(
                    new[] { 2 * embedDim }));
                this.patchMergeNormWeight = new PradOp(new Tensor(
                    new[] { 4 * embedDim }));
                this.patchMergeNormBias = new PradOp(new Tensor(
                    new[] { 4 * embedDim }));
            }

            // Initialize transformer blocks
            this.blocks = new List<SwinTransformerBlock>();
            for (int i = 0; i < depth; i++)
            {
                this.blocks.Add(new SwinTransformerBlock(
                    tools,
                    embedDim,
                    numHeads,
                    windowSize,
                    mlpRatio,
                    shifted: (i % 2 == 1)  // Alternate between regular and shifted
                ));
            }
        }

        /// <summary>
        /// Applies cyclic shift to the input feature map.
        /// </summary>
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
        /// Forward pass through the stage.
        /// </summary>
        public PradResult Forward(PradOp input)
        {
            var x = input;

            // Apply patch merging if downsampling
            if (this.downsample)
            {
                x = this.PatchMerging(x);
            }

            // Process through transformer blocks
            foreach (var block in this.blocks)
            {
                x = block.Forward(x.PradOp);
            }

            return x;
        }

        private PradResult PatchMerging(PradOp x)
        {
            var shape = x.CurrentShape;
            var B = shape[0];
            var H = shape[1];
            var W = shape[2];
            var C = shape[3];

            // Ensure dimensions are even
            if (H % 2 != 0 || W % 2 != 0)
                throw new ArgumentException("Feature map dimensions must be even for patch merging");

            // Get the four groups of patches
            var x0 = x.Indexer(":", "::2", "::2", ":");     // Top-left
            var x1 = x.Indexer(":", "::2", "1::2", ":");    // Top-right
            var x2 = x.Indexer(":", "1::2", "::2", ":");    // Bottom-left
            var x3 = x.Indexer(":", "1::2", "1::2", ":");   // Bottom-right

            // Concatenate along channel dimension
            var concatenated = x0.PradOp.Concat(
                new[] { x1.Result, x2.Result, x3.Result },
                axis: -1);

            // Layer normalization
            var normalized = concatenated.Then(result => result.PradOp.LayerNorm(1e-5))
                .Then(PradOp.MulOp, this.patchMergeNormWeight.CurrentTensor)
                .Then(PradOp.AddOp, this.patchMergeNormBias.CurrentTensor);

            // Linear transformation
            return normalized.Then(PradOp.MatMulOp, this.patchMergeWeight.CurrentTensor)
                .Then(PradOp.AddOp, this.patchMergeBias.CurrentTensor);
        }
    }

    /// <summary>
    /// Utility class for computing efficient attention masks for shifted windows.
    /// </summary>
    public class WindowMaskGenerator
    {
        private readonly int windowSize;
        private readonly int shift;
        private Dictionary<(int, int), Tensor> maskCache;

        public WindowMaskGenerator(int windowSize, int shift)
        {
            this.windowSize = windowSize;
            this.shift = shift;
            this.maskCache = new Dictionary<(int, int), Tensor>();
        }

        public Tensor GetAttentionMask(int height, int width)
        {
            var key = (height, width);
            if (this.maskCache.ContainsKey(key))
                return this.maskCache[key];

            var mask = GenerateAttentionMask(height, width);
            this.maskCache[key] = mask;
            return mask;
        }

        private Tensor GenerateAttentionMask(int height, int width)
        {
            // Calculate number of windows
            var h_slices = (height - this.windowSize) // shifted
                        / (this.windowSize - this.shift) + 2;
            var w_slices = (width - this.windowSize)  // shifted
                        / (this.windowSize - this.shift) + 2;

            var mask = new float[h_slices * this.windowSize, w_slices * this.windowSize];

            // Fill mask values
            for (int h = 0; h < h_slices * this.windowSize; h++)
            {
                var h_window = h / this.windowSize;
                for (int w = 0; w < w_slices * this.windowSize; w++)
                {
                    var w_window = w / this.windowSize;
                    var valid = h < height && w < width;
                    mask[h, w] = valid ? 0 : float.NegativeInfinity;
                }
            }

            return new Tensor(new[] { h_slices * this.windowSize, w_slices * this.windowSize }, mask);
        }
    }
}