public partial class PradSwinTransformerTools
{
    /// <summary>
    /// Implements adaptive average pooling for the Swin Transformer.
    /// </summary>
    public class AdaptiveAveragePool
    {
        private readonly int[] outputSize;

        /// <summary>
        /// Initializes a new instance of the <see cref="AdaptiveAveragePool"/> class.
        /// </summary>
        /// <param name="outputSize">Desired output size [H, W]. Use [1, 1] for global average pooling.</param>
        public AdaptiveAveragePool(int[] outputSize)
        {
            if (outputSize.Length != 2)
                throw new ArgumentException("Output size must be [H, W]");
            this.outputSize = outputSize;
        }

        /// <summary>
        /// Applies adaptive average pooling to the input tensor.
        /// </summary>
        /// <param name="input">Input tensor of shape [batch_size, height, width, channels].</param>
        /// <returns>Pooled tensor of shape [batch_size, output_height, output_width, channels].</returns>
        public PradResult Forward(PradOp input)
        {
            var shape = input.CurrentShape;
            var batchSize = shape[0];
            var inputHeight = shape[1];
            var inputWidth = shape[2];
            var channels = shape[3];

            var outputHeight = this.outputSize[0];
            var outputWidth = this.outputSize[1];

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
                                    float sum = 0;
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
    }

    /// <summary>
    /// Enhanced classification head for Swin Transformer.
    /// </summary>
    public class SwinClassificationHead
    {
        private readonly PradSwinTransformerTools tools;
        private readonly int numClasses;
        private readonly int embedDim;
        private readonly AdaptiveAveragePool pooling;
        private readonly PradOp normWeight;
        private readonly PradOp normBias;
        private readonly PradOp classifierWeight;
        private readonly PradOp classifierBias;

        /// <summary>
        /// Initializes a new instance of the <see cref="SwinClassificationHead"/> class.
        /// </summary>
        public SwinClassificationHead(
            PradSwinTransformerTools tools,
            int numClasses,
            int embedDim)
        {
            this.tools = tools;
            this.numClasses = numClasses;
            this.embedDim = embedDim;
            
            // Initialize adaptive average pooling to 1x1
            this.pooling = new AdaptiveAveragePool(new[] { 1, 1 });

            // Initialize layer norm parameters
            this.normWeight = new PradOp(new Tensor(new[] { embedDim }, 1.0f));
            this.normBias = new PradOp(new Tensor(new[] { embedDim }));

            // Initialize classifier parameters
            this.classifierWeight = new PradOp(new Tensor(new[] { embedDim, numClasses }));
            this.classifierBias = new PradOp(new Tensor(new[] { numClasses }));

            InitializeWeights();
        }

        /// <summary>
        /// Forward pass of the classification head.
        /// </summary>
        public PradResult Forward(PradOp input)
        {
            // Global average pooling
            var pooled = this.pooling.Forward(input);

            // Reshape to [batch_size, embedDim]
            var flattened = pooled.Then(PradOp.ReshapeOp, new[] { input.CurrentShape[0], this.embedDim });

            // Layer normalization
            var normalized = flattened.Then(result => result.PradOp.LayerNorm(1e-5))
                .Then(PradOp.MulOp, this.normWeight.CurrentTensor)
                .Then(PradOp.AddOp, this.normBias.CurrentTensor);

            // Classification layer
            return normalized.Then(PradOp.MatMulOp, this.classifierWeight.CurrentTensor)
                .Then(PradOp.AddOp, this.classifierBias.CurrentTensor);
        }

        private void InitializeWeights()
        {
            // Initialize classifier weights using truncated normal distribution
            var random = new Random(42);
            var stddev = 0.02f;
            
            var classifierData = new float[this.embedDim * this.numClasses];
            for (int i = 0; i < classifierData.Length; i++)
            {
                // Box-Muller transform
                var u1 = random.NextDouble();
                var u2 = random.NextDouble();
                var z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                
                // Truncate at 2 standard deviations
                z = Math.Max(Math.Min(z * stddev, 2 * stddev), -2 * stddev);
                classifierData[i] = (float)z;
            }

            this.classifierWeight.Reset(new Tensor(new[] { this.embedDim, this.numClasses }, classifierData));
        }
    }

    /// <summary>
    /// Example usage of adaptive pooling and classification head.
    /// </summary>
    public void DemonstrateClassification()
    {
        // Create sample input tensor [batch_size, height, width, channels]
        var batchSize = 32;
        var height = 7;  // Final stage output
        var width = 7;   // Final stage output
        var channels = 768;  // Final embedding dimension

        var input = new PradOp(new Tensor(new[] { batchSize, height, width, channels }));

        // Create classification head
        var classHead = new SwinClassificationHead(this, numClasses: 1000, embedDim: channels);

        // Forward pass
        var logits = classHead.Forward(input);

        // Optional: Apply softmax for probabilities
        var probabilities = logits.Then(op => op.PradOp.Softmax(-1));

        Console.WriteLine($"Classification output shape: [{string.Join(", ", probabilities.PradOp.CurrentShape)}]");
    }
}