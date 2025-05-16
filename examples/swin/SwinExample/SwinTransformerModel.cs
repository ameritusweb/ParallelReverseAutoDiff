namespace SwinExample
{
    using ParallelReverseAutoDiff.RMAD;

    public class SwinTransformerModel
    {
        private readonly IModelLayer patchEmbeddingLayer;
        private readonly IModelLayer[] stageTransformerLayers;
        private readonly IModelLayer[] patchMergingLayers;
        private readonly IModelLayer classificationLayer;

        private readonly SwinTransformerNetwork network;

        private readonly int[] stageDepths;
        private readonly int[] numHeads;
        private readonly int[] windowSizes;
        private readonly int[] embedDims;
        private readonly int numClasses;
        private readonly int patchSize;
        private readonly float mlpRatio;
        private readonly bool qkvBias;
        private readonly float[] dropPaths;

        public SwinTransformerModel(
            SwinTransformerNetwork network,
            int imageSize,
            int numClasses,
            int[] stageDepths = null,
            int[] numHeads = null,
            int[] windowSizes = null,
            int[] embedDims = null,
            int patchSize = 4,
            float mlpRatio = 4.0f,
            bool qkvBias = true,
            float[] dropPaths = null)
        {
            this.network = network;
            // Default configuration following the original Swin-T architecture
            this.stageDepths = stageDepths ?? new[] { 2, 2, 6, 2 };
            this.numHeads = numHeads ?? new[] { 3, 6, 12, 24 };
            this.windowSizes = windowSizes ?? new[] { 7, 7, 7, 7 };
            this.embedDims = embedDims ?? new[] { 96, 192, 384, 768 };
            this.numClasses = numClasses;
            this.patchSize = patchSize;
            this.mlpRatio = mlpRatio;
            this.qkvBias = qkvBias;
            this.dropPaths = dropPaths ?? new[] { 0.0f, 0.0f, 0.0f, 0.0f };

            // Validate input dimensions
            ValidateArchitectureParameters(imageSize);

            // Initialize all layers
            this.patchEmbeddingLayer = BuildPatchEmbeddingLayer();
            this.stageTransformerLayers = BuildTransformerStages();
            this.patchMergingLayers = BuildPatchMergingLayers();
            this.classificationLayer = BuildClassificationLayer();
        }

        private IModelLayer BuildPatchEmbeddingLayer()
        {
            var builder = new ModelLayerBuilder(this.network)
                // Patch embedding projection: from (patch_size * patch_size * 3) to embed_dim[0]
                .AddModelElementGroup(
                    "patch_embed_proj",
                    new[] { this.embedDims[0], this.patchSize * this.patchSize * 3 },
                    InitializationType.Xavier)
                .AddModelElementGroup(
                    "patch_embed_norm_weight",
                    new[] { this.embedDims[0] },
                    InitializationType.Ones)
                .AddModelElementGroup(
                    "patch_embed_norm_bias",
                    new[] { this.embedDims[0] },
                    InitializationType.Zeroes);

            return builder.Build();
        }

        private IModelLayer[] BuildTransformerStages()
        {
            var stages = new IModelLayer[this.stageDepths.Length];

            for (int stageIdx = 0; stageIdx < this.stageDepths.Length; stageIdx++)
            {
                var dimModel = this.embedDims[stageIdx];
                var numHead = this.numHeads[stageIdx];
                var windowSize = this.windowSizes[stageIdx];
                var depth = this.stageDepths[stageIdx];

                var stageBuilder = new ModelLayerBuilder(this.network);

                // Build blocks for this stage
                for (int blockIdx = 0; blockIdx < depth; blockIdx++)
                {
                    var blockPrefix = $"stage_{stageIdx}_block_{blockIdx}";
                    var isShiftedBlock = blockIdx % 2 == 1;

                    // QKV projection weights and bias
                    stageBuilder.AddModelElementGroup(
                        $"{blockPrefix}_qkv_weight",
                        new[] { dimModel, 3 * dimModel },
                        InitializationType.Xavier);

                    if (this.qkvBias)
                    {
                        stageBuilder.AddModelElementGroup(
                            $"{blockPrefix}_qkv_bias",
                            new[] { 3 * dimModel },
                            InitializationType.Zeroes);
                    }

                    // Attention output projection
                    stageBuilder.AddModelElementGroup(
                        $"{blockPrefix}_proj_weight",
                        new[] { dimModel, dimModel },
                        InitializationType.Xavier)
                        .AddModelElementGroup(
                            $"{blockPrefix}_proj_bias",
                            new[] { dimModel },
                            InitializationType.Zeroes);

                    // Layer normalization parameters
                    stageBuilder.AddModelElementGroup(
                        $"{blockPrefix}_norm1_weight",
                        new[] { dimModel },
                        InitializationType.Ones)
                        .AddModelElementGroup(
                            $"{blockPrefix}_norm1_bias",
                            new[] { dimModel },
                            InitializationType.Zeroes)
                        .AddModelElementGroup(
                            $"{blockPrefix}_norm2_weight",
                            new[] { dimModel },
                            InitializationType.Ones)
                        .AddModelElementGroup(
                            $"{blockPrefix}_norm2_bias",
                            new[] { dimModel },
                            InitializationType.Zeroes);

                    // MLP weights and biases
                    var mlpHiddenDim = (int)(dimModel * this.mlpRatio);
                    stageBuilder.AddModelElementGroup(
                        $"{blockPrefix}_mlp_fc1_weight",
                        new[] { dimModel, mlpHiddenDim },
                        InitializationType.Xavier)
                        .AddModelElementGroup(
                            $"{blockPrefix}_mlp_fc1_bias",
                            new[] { mlpHiddenDim },
                            InitializationType.Zeroes)
                        .AddModelElementGroup(
                            $"{blockPrefix}_mlp_fc2_weight",
                            new[] { mlpHiddenDim, dimModel },
                            InitializationType.Xavier)
                        .AddModelElementGroup(
                            $"{blockPrefix}_mlp_fc2_bias",
                            new[] { dimModel },
                            InitializationType.Zeroes);

                    // Relative position bias table
                    var numRelativeDistance = (2 * windowSize - 1) * (2 * windowSize - 1);
                    stageBuilder.AddModelElementGroup(
                        $"{blockPrefix}_relative_position_bias_table",
                        new[] { numRelativeDistance, numHead },
                        InitializationType.Xavier);
                }

                stages[stageIdx] = stageBuilder.Build();
            }

            return stages;
        }

        private IModelLayer[] BuildPatchMergingLayers()
        {
            var mergingLayers = new IModelLayer[this.stageDepths.Length - 1];

            for (int i = 0; i < this.stageDepths.Length - 1; i++)
            {
                var inputDim = this.embedDims[i];
                var outputDim = this.embedDims[i + 1];

                var builder = new ModelLayerBuilder(this.network)
                    // Merging projection: from 4 * input_dim to output_dim
                    .AddModelElementGroup(
                        $"merge_{i}_proj_weight",
                        new[] { outputDim, 4 * inputDim },
                        InitializationType.Xavier)
                    .AddModelElementGroup(
                        $"merge_{i}_norm_weight",
                        new[] { 4 * inputDim },
                        InitializationType.Ones)
                    .AddModelElementGroup(
                        $"merge_{i}_norm_bias",
                        new[] { 4 * inputDim },
                        InitializationType.Zeroes);

                mergingLayers[i] = builder.Build();
            }

            return mergingLayers;
        }

        private IModelLayer BuildClassificationLayer()
        {
            var finalDim = this.embedDims[^1];

            var builder = new ModelLayerBuilder(this.network)
                // Layer normalization for final features
                .AddModelElementGroup(
                    "final_norm_weight",
                    new[] { finalDim },
                    InitializationType.Ones)
                .AddModelElementGroup(
                    "final_norm_bias",
                    new[] { finalDim },
                    InitializationType.Zeroes)
                // Classification head
                .AddModelElementGroup(
                    "classifier_weight",
                    new[] { this.numClasses, finalDim },
                    InitializationType.Xavier)
                .AddModelElementGroup(
                    "classifier_bias",
                    new[] { this.numClasses },
                    InitializationType.Zeroes);

            return builder.Build();
        }

        private void ValidateArchitectureParameters(int imageSize)
        {
            if (imageSize % this.patchSize != 0)
            {
                throw new ArgumentException($"Image size {imageSize} must be divisible by patch size {this.patchSize}");
            }

            if (this.stageDepths.Length != 4 ||
                this.numHeads.Length != 4 ||
                this.windowSizes.Length != 4 ||
                this.embedDims.Length != 4)
            {
                throw new ArgumentException("Stage parameters must have length 4");
            }

            // Validate that image size can be properly downsampled through all stages
            var featureSize = imageSize / this.patchSize;
            for (int i = 0; i < 3; i++)
            {
                if (featureSize % 2 != 0)
                {
                    throw new ArgumentException($"Feature map size {featureSize} at stage {i} must be even");
                }
                featureSize /= 2;
            }

            // Validate window sizes
            foreach (var windowSize in this.windowSizes)
            {
                if (windowSize <= 0)
                {
                    throw new ArgumentException("Window size must be positive");
                }
            }
        }

        // Helper method to get layer weights
        public Matrix GetWeight(string stage, string block, string parameter)
        {
            if (stage == "patch_embedding")
            {
                return this.patchEmbeddingLayer.WeightMatrix(parameter);
            }
            else if (stage == "classification")
            {
                return this.classificationLayer.WeightMatrix(parameter);
            }
            else if (stage.StartsWith("stage_"))
            {
                var stageIdx = int.Parse(stage.Split('_')[1]);
                return this.stageTransformerLayers[stageIdx].WeightMatrix($"{block}_{parameter}");
            }
            else if (stage.StartsWith("merge_"))
            {
                var mergeIdx = int.Parse(stage.Split('_')[1]);
                return this.patchMergingLayers[mergeIdx].WeightMatrix(parameter);
            }

            throw new ArgumentException($"Invalid stage: {stage}");
        }

        // Helper method to get layer gradients
        public Matrix GetGradient(string stage, string block, string parameter)
        {
            if (stage == "patch_embedding")
            {
                return this.patchEmbeddingLayer.GradientMatrix(parameter);
            }
            else if (stage == "classification")
            {
                return this.classificationLayer.GradientMatrix(parameter);
            }
            else if (stage.StartsWith("stage_"))
            {
                var stageIdx = int.Parse(stage.Split('_')[1]);
                return this.stageTransformerLayers[stageIdx].GradientMatrix($"{block}_{parameter}");
            }
            else if (stage.StartsWith("merge_"))
            {
                var mergeIdx = int.Parse(stage.Split('_')[1]);
                return this.patchMergingLayers[mergeIdx].GradientMatrix(parameter);
            }

            throw new ArgumentException($"Invalid stage: {stage}");
        }
    }
}
