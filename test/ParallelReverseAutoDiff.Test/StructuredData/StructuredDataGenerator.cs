namespace ParallelReverseAutoDiff.Test.StructuredData
{
    using ParallelReverseAutoDiff.PRAD;
    using ParallelReverseAutoDiff.PRAD.Extensions;

    public class StructuredDataGenerator
    {
        private readonly int embeddingDim;
        private readonly int outputRows;
        private readonly int outputCols;
        private readonly double learningRate;

        private PradOp embeddingTable;  // The learned embeddings
        private PradOp projectionLayer; // Projects embedding to larger dimension
        private PradOp structureLayer;  // Generates structured output
        private PradOp gamma;           // Scale parameter for layer norm
        private PradOp beta;            // Shift parameter for layer norm

        public StructuredDataGenerator(int embeddingDim, int numClasses, int outputRows, int outputCols, double learningRate = 0.001)
        {
            this.embeddingDim = embeddingDim;
            this.outputRows = outputRows;
            this.outputCols = outputCols;
            this.learningRate = learningRate;

            // Initialize embedding table with random values
            var embeddings = Tensor.RandomUniform(new[] { numClasses, embeddingDim }, -0.1, 0.1);
            this.embeddingTable = new PradOp(embeddings);

            // Initialize projection layer with normalized weights
            var projectionWeights = Tensor.RandomUniform(new[] { embeddingDim, outputRows * outputCols / 4 }, -0.1, 0.1);
            this.projectionLayer = new PradOp(projectionWeights);

            // Initialize structure layer
            var structureWeights = Tensor.RandomUniform(new[] { outputRows * outputCols / 4, outputRows * outputCols }, -0.1, 0.1);
            this.structureLayer = new PradOp(structureWeights);

            // Initialize normalization parameters
            this.gamma = new PradOp(new Tensor(new[] { outputRows * outputCols / 4 }, 1.0));
            this.beta = new PradOp(new Tensor(new[] { outputRows * outputCols / 4 }, 0.0));
        }

        public (PradResult Output, PradResult TotalLoss) Forward(int classIndex)
        {
            // Get embedding for class
            var embedding = this.embeddingTable.Indexer($"{classIndex}:", ":");

            // Project to larger dimension with weight normalization and GLU
            var projected = embedding.PradOp.MatMul(this.projectionLayer.CurrentTensor);
            var normalizedProjection = projected.PradOp.LayerNorm();
            var gluOutput = normalizedProjection.PradOp.GLU();

            // Add noise for regularization
            var noisyOutput = gluOutput.PradOp.GaussianNoise(0.0, 0.01);

            // Apply dropout
            var droppedOutput = noisyOutput.PradOp.Dropout(0.1);

            // Generate structured output with layer normalization
            var structureOutput = droppedOutput.Then(PradOp.MatMulOp, this.structureLayer.CurrentTensor);
            var normalizedStructure = structureOutput.PradOp.BatchNorm(this.gamma, this.beta);

            // Apply SymmetricSoftmax activation
            var activated = normalizedStructure.PradOp.SymmetricSoftmax(temperature: 1.0);

            // Reshape to desired output dimensions
            var reshaped = activated.Then(PradOp.ReshapeOp, new[] { this.outputRows, this.outputCols });

            // Calculate regularization losses
            var totalVariationLoss = reshaped.PradOp.TotalVariation();

            // Create frequency weight matrix for spectral regularization
            var freqWeights = new Tensor(reshaped.Result.Shape);
            for (int i = 0; i < this.outputRows; i++)
            {
                for (int j = 0; j < this.outputCols; j++)
                {
                    var freqI = i < this.outputRows / 2 ? i : this.outputRows - i;
                    var freqJ = j < this.outputCols / 2 ? j : this.outputCols - j;
                    freqWeights[i, j] = Math.Sqrt(freqI * freqI + freqJ * freqJ);
                }
            }
            var spectralLoss = reshaped.PradOp.SpectralRegularize(new PradOp(freqWeights));

            // Combine losses
            var totalLoss = totalVariationLoss.PradOp.Mul(new Tensor(totalVariationLoss.PradOp.CurrentShape, 0.5))
                .Then(PradOp.AddOp, spectralLoss.PradOp.Mul(new Tensor(spectralLoss.PradOp.CurrentShape, 0.3)).Result);

            return (reshaped, totalLoss);
        }

        public void Update()
        {
            // Apply gradient clipping
            var clipper = PradClipper.CreateGradientClipper();
            clipper.ClipGradients(this.embeddingTable.SeedGradient);
            clipper.ClipGradients(this.projectionLayer.SeedGradient);
            clipper.ClipGradients(this.structureLayer.SeedGradient);
            clipper.ClipGradients(this.gamma.SeedGradient);
            clipper.ClipGradients(this.beta.SeedGradient);

            // Initialize Adam optimizers
            var optimizerEmbed = new AdamOptimizer(this.learningRate);
            var optimizerProj = new AdamOptimizer(this.learningRate);
            var optimizerStruct = new AdamOptimizer(this.learningRate);
            var optimizerGamma = new AdamOptimizer(this.learningRate);
            var optimizerBeta = new AdamOptimizer(this.learningRate);

            optimizerEmbed.Initialize(this.embeddingTable.CurrentTensor);
            optimizerProj.Initialize(this.projectionLayer.CurrentTensor);
            optimizerStruct.Initialize(this.structureLayer.CurrentTensor);
            optimizerGamma.Initialize(this.gamma.CurrentTensor);
            optimizerBeta.Initialize(this.beta.CurrentTensor);

            // Update weights
            optimizerEmbed.UpdateWeights(this.embeddingTable.CurrentTensor, this.embeddingTable.SeedGradient);
            optimizerProj.UpdateWeights(this.projectionLayer.CurrentTensor, this.projectionLayer.SeedGradient);
            optimizerStruct.UpdateWeights(this.structureLayer.CurrentTensor, this.structureLayer.SeedGradient);
            optimizerGamma.UpdateWeights(this.gamma.CurrentTensor, this.gamma.SeedGradient);
            optimizerBeta.UpdateWeights(this.beta.CurrentTensor, this.beta.SeedGradient);
        }
    }
}
