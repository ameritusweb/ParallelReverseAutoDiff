using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.PRAD.Extensions;

namespace SwinExample
{
    public class SwinTransformerTools
    {
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
