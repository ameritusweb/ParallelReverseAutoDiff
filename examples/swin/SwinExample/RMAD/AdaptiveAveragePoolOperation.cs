using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.RMAD;

namespace SwinExample.RMAD
{
    /// <summary>
    /// Operation that performs adaptive average pooling, converting features to target size.
    /// </summary>
    public class AdaptiveAveragePoolOperation : Operation
    {
        private PradOp input;
        private PradOp resultOp;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new AdaptiveAveragePoolOperation();
        }

        /// <summary>
        /// Performs the forward operation for adaptive average pooling.
        /// </summary>
        /// <param name="input">Input tensor of shape [B, H, W, C]</param>
        /// <param name="options">Options tensor containing [outputHeight, outputWidth]</param>
        /// <returns>Output tensor of shape [B, outputHeight, outputWidth, C]</returns>
        public Matrix Forward(Matrix input, Matrix options)
        {
            SwinTransformerTools tools = new SwinTransformerTools();
            
            // Convert inputs to PradOp
            this.input = new PradOp(input.ToTensor());
            var optionsPradOp = new PradOp(options.ToTensor());

            // Perform adaptive average pooling
            var result = tools.AdaptiveAveragePool(this.input, optionsPradOp);
            
            this.resultOp = result.PradOp;
            this.Output = result.Result.ToMatrix();
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            // Perform backward pass using PradOp
            this.resultOp.Back(dLdOutput.ToTensor());

            // Get gradient with respect to input
            var dInput = this.input.SeedGradient.ToMatrix();

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .Build();
        }
    }
}