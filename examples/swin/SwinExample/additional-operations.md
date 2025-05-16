using ParallelReverseAutoDiff.PRAD;

namespace SwinExample
{
    public static class AdditionalSwinOperations
    {
        /// <summary>
        /// Implements cyclic shift for shifted window attention.
        /// </summary>
        public class CyclicShiftOperation
        {
            private readonly int shift;

            public CyclicShiftOperation(int shift)
            {
                this.shift = shift;
            }

            public PradResult Forward(PradOp input)
            {
                if (this.shift == 0)
                {
                    return input.NoOp();
                }

                var shape = input.CurrentShape;
                var B = shape[0];
                var H = shape[1];
                var W = shape[2];
                var C = shape[3];

                // Roll along H dimension
                var sliceH = input.Indexer($":", $"{H-this.shift}:", ":", ":");
                var remainderH = input.Indexer($":", $":{H-this.shift}", ":", ":");
                var rolledH = sliceH.PradOp.Concat(new[] { remainderH.Result }, axis: 1);

                // Roll along W dimension
                var sliceW = rolledH.Then(x => x.PradOp.Indexer($":", ":", $"{W-this.shift}:", ":"));
                var remainderW = rolledH.Then(x => x.PradOp.Indexer($":", ":", $":{W-this.shift}", ":"));
                
                return sliceW.PradOp.Concat(new[] { remainderW.Result }, axis: 2);
            }
        }

        /// <summary>
        /// Implements patch merging for downsampling between stages.
        /// </summary>
        public class PatchMergingOperation
        {
            private readonly int dimModel;

            public PatchMergingOperation(int dimModel)
            {
                this.dimModel = dimModel;
            }

            public PradResult Forward(PradOp input)
            {
                var shape = input.CurrentShape;
                var B = shape[0];
                var H = shape[1];
                var W = shape[2];
                var C = shape[3];

                if (H % 2 != 0 || W % 2 != 0)
                {
                    throw new ArgumentException("Feature map dimensions must be even for patch merging");
                }

                // Get the four groups of patches
                var x0 = input.Indexer(":", "::2", "::2", ":");     // Top-left
                var x1 = input.Indexer(":", "::2", "1::2", ":");    // Top-right
                var x2 = input.Indexer(":", "1::2", "::2", ":");    // Bottom-left
                var x3 = input.Indexer(":", "1::2", "1::2", ":");   // Bottom-right

                // Concatenate along channel dimension
                var concatenated = x0.PradOp.Concat(
                    new[] { x1.Result, x2.Result, x3.Result },
                    axis: -1);

                // Reshape to [B, H/2 * W/2, 4*C]
                return concatenated.Then(PradOp.ReshapeOp, new[] { B, H / 2 * W / 2, 4 * C });
            }
        }

        /// <summary>
        /// Implements patch extraction from input images.
        /// </summary>
        public class PatchExtractionOperation
        {
            private readonly int patchSize;
            private readonly int inChannels;

            public PatchExtractionOperation(int patchSize, int inChannels)
            {
                this.patchSize = patchSize;
                this.inChannels = inChannels;
            }

            public PradResult Forward(PradOp input)
            {
                var shape = input.CurrentShape;
                var B = shape[0];
                var H = shape[1];
                var W = shape[2];

                if (H % this.patchSize != 0 || W % this.patchSize != 0)
                {
                    throw new ArgumentException("Image dimensions must be divisible by patch size");
                }

                // Extract patches using convolution-like operation
                var patches = input.ExtractPatches(
                    filterSize: new[] { this.patchSize, this.patchSize },
                    strides: new[] { this.patchSize, this.patchSize },
                    padding: "VALID");

                // Reshape to [B, num_patches, patch_size * patch_size * channels]
                var numPatchesH = H / this.patchSize;
                var numPatchesW = W / this.patchSize;
                return patches.Then(PradOp.ReshapeOp, 
                    new[] { B, numPatchesH * numPatchesW, this.patchSize * this.patchSize * this.inChannels });
            }
        }

        /// <summary>
        /// Implements drop path (stochastic depth) for regularization.
        /// </summary>
        public class DropPathOperation
        {
            private readonly float dropPath;
            private readonly bool training;

            public DropPathOperation(float dropPath, bool training = true)
            {
                this.dropPath = dropPath;
                this.training = training;
            }

            public PradResult Forward(PradOp input)
            {
                if (!this.training || this.dropPath == 0.0f)
                {
                    return input.NoOp();
                }

                var shape = input.CurrentShape;
                var keepProb = 1.0f - this.dropPath;

                // Generate random mask with shape [B, 1, 1, 1]
                var maskShape = new[] { shape[0], 1, 1, 1 };
                var random = new Random();
                var maskData = new float[shape[0]];
                for (int i = 0; i < shape[0]; i++)
                {
                    maskData[i] = random.NextSingle() < keepProb ? 1.0f / keepProb : 0.0f;
                }
                var mask = new Tensor(maskShape, maskData);

                // Apply mask
                return input.Mul(mask);
            }
        }

        /// <summary>
        /// Implements the MLP block used in Swin Transformer.
        /// </summary>
        public class MLPOperation
        {
            private readonly int dimHidden;
            private readonly float dropoutRate;
            private readonly bool training;

            public MLPOperation(int dimHidden, float dropoutRate = 0.0f, bool training = true)
            {
                this.dimHidden = dimHidden;
                this.dropoutRate = dropoutRate;
                this.training = training;
            }

            public PradResult Forward(PradOp input, PradOp fc1Weight, PradOp fc1Bias, 
                                    PradOp fc2Weight, PradOp fc2Bias)
            {
                // First linear layer
                var fc1 = input.MatMul(fc1Weight.CurrentTensor)
                              .Then(PradOp.AddOp, fc1Bias.CurrentTensor);

                // GELU activation
                var geluOp = new SwinTransformerOperations.GELUOperation();
                var activated = geluOp.Forward(fc1.PradOp);

                // Dropout after activation
                var activatedDropout = this.training && this.dropoutRate > 0
                    ? activated.Then(x => x.PradOp.Dropout(this.dropoutRate))
                    : activated;

                // Second linear layer
                var fc2 = activatedDropout.Then(PradOp.MatMulOp, fc2Weight.CurrentTensor)
                                        .Then(PradOp.AddOp, fc2Bias.CurrentTensor);

                // Final dropout
                return this.training && this.dropoutRate > 0
                    ? fc2.Then(x => x.PradOp.Dropout(this.dropoutRate))
                    : fc2;
            }
        }
    }
}
