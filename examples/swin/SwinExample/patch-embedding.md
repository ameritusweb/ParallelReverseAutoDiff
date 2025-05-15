using System;
using ParallelReverseAutoDiff.PRAD;

namespace ParallelReverseAutoDiff.PRAD.SwinTransformer
{
    /// <summary>
    /// Implements patch embedding for Swin Transformer.
    /// Converts input images to patches and applies linear embedding.
    /// </summary>
    public class PatchEmbedding
    {
        private readonly int patchSize;
        private readonly int embedDim;
        private readonly bool normalize;
        private readonly PradOp projectionWeights;
        private readonly PradOp projectionBias;
        private readonly PradOp normWeights; // Optional layer norm
        private readonly PradOp normBias;    // Optional layer norm

        /// <summary>
        /// Initializes a new instance of the <see cref="PatchEmbedding"/> class.
        /// </summary>
        /// <param name="inChannels">Number of input channels (e.g., 3 for RGB).</param>
        /// <param name="embedDim">Embedding dimension.</param>
        /// <param name="patchSize">Size of each patch (e.g., 4 for 4x4 patches).</param>
        /// <param name="normalize">Whether to apply layer normalization.</param>
        public PatchEmbedding(int inChannels, int embedDim, int patchSize = 4, bool normalize = true)
        {
            this.patchSize = patchSize;
            this.embedDim = embedDim;
            this.normalize = normalize;

            // Initialize projection weights and bias
            // Shape: [patch_size * patch_size * in_channels, embed_dim]
            var projWeightsData = new float[patchSize * patchSize * inChannels * embedDim];
            var random = new Random(42); // Fixed seed for reproducibility
            for (int i = 0; i < projWeightsData.Length; i++)
            {
                // Xavier initialization
                var limit = Math.Sqrt(6.0 / (patchSize * patchSize * inChannels + embedDim));
                projWeightsData[i] = (float)((random.NextDouble() * 2 * limit) - limit);
            }

            this.projectionWeights = new PradOp(new Tensor(
                new[] { patchSize * patchSize * inChannels, embedDim },
                projWeightsData));

            // Initialize bias
            var biasData = new float[embedDim];
            this.projectionBias = new PradOp(new Tensor(new[] { 1, embedDim }, biasData));

            if (normalize)
            {
                // Initialize layer norm parameters
                this.normWeights = new PradOp(new Tensor(new[] { 1, embedDim }, 1.0f));
                this.normBias = new PradOp(new Tensor(new[] { 1, embedDim }, 0.0f));
            }
        }

        /// <summary>
        /// Applies patch embedding to the input tensor.
        /// </summary>
        /// <param name="input">Input tensor of shape [batch_size, height, width, channels].</param>
        /// <returns>Embedded patches of shape [batch_size, num_patches_h, num_patches_w, embed_dim].</returns>
        public PradResult Forward(PradOp input)
        {
            var inputShape = input.CurrentShape;
            var batchSize = inputShape[0];
            var height = inputShape[1];
            var width = inputShape[2];
            var channels = inputShape[3];

            // Ensure input dimensions are divisible by patch size
            if (height % this.patchSize != 0 || width % this.patchSize != 0)
            {
                throw new ArgumentException($"Input height ({height}) and width ({width}) must be divisible by patch size ({this.patchSize})");
            }

            // Extract patches using the custom patches operation
            // This creates patches of size [patchSize x patchSize] with stride equal to patchSize
            var patches = input.ExtractPatches(
                filterSize: new[] { this.patchSize, this.patchSize },
                strides: new[] { this.patchSize, this.patchSize },
                padding: "VALID");

            // Reshape patches to [batch_size * num_patches, patch_size * patch_size * channels]
            var numPatchesH = height / this.patchSize;
            var numPatchesW = width / this.patchSize;
            var patchesReshaped = patches.Then(PradOp.ReshapeOp, 
                new[] { batchSize * numPatchesH * numPatchesW, this.patchSize * this.patchSize * channels });

            // Apply linear projection
            var projected = patchesReshaped.Then(PradOp.MatMulOp, this.projectionWeights.CurrentTensor);
            var biased = projected.Then(PradOp.AddOp, this.projectionBias.CurrentTensor);

            // Reshape back to [batch_size, num_patches_h, num_patches_w, embed_dim]
            var output = biased.Then(PradOp.ReshapeOp, new[] { batchSize, numPatchesH, numPatchesW, this.embedDim });

            if (this.normalize)
            {
                // Apply layer normalization
                output = output.Then(result => result.PradOp.LayerNorm(1e-5))
                    .Then(PradOp.MulOp, this.normWeights.CurrentTensor)
                    .Then(PradOp.AddOp, this.normBias.CurrentTensor);
            }

            return output;
        }

        /// <summary>
        /// Gets the projection weights for external access.
        /// </summary>
        /// <returns>The projection weights PradOp.</returns>
        public PradOp GetProjectionWeights() => this.projectionWeights;

        /// <summary>
        /// Gets the projection bias for external access.
        /// </summary>
        /// <returns>The projection bias PradOp.</returns>
        public PradOp GetProjectionBias() => this.projectionBias;

        /// <summary>
        /// Gets the normalization weights if normalization is enabled.
        /// </summary>
        /// <returns>The normalization weights PradOp or null if normalization is disabled.</returns>
        public PradOp GetNormalizationWeights() => this.normalize ? this.normWeights : null;

        /// <summary>
        /// Gets the normalization bias if normalization is enabled.
        /// </summary>
        /// <returns>The normalization bias PradOp or null if normalization is disabled.</returns>
        public PradOp GetNormalizationBias() => this.normalize ? this.normBias : null;
    }
}