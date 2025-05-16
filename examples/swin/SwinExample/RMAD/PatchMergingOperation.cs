using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.RMAD;

namespace SwinExample.RMAD
{
    /// <summary>
    /// Operation that performs patch merging to reduce resolution and increase channel dimension.
    /// </summary>
    public class PatchMergingOperation : Operation
    {
        private PradOp input;
        private PradOp mergeWeight;
        private PradOp mergeBias;
        private PradOp normWeight;
        private PradOp normBias;
        private PradOp resultOp;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new PatchMergingOperation();
        }

        /// <summary>
        /// Performs the forward operation for patch merging.
        /// </summary>
        /// <param name="input">Input tensor of shape [B, H, W, C]</param>
        /// <param name="mergeWeight">Weight matrix for linear transformation</param>
        /// <param name="mergeBias">Bias for linear transformation</param>
        /// <param name="normWeight">Weight for layer normalization</param>
        /// <param name="normBias">Bias for layer normalization</param>
        /// <returns>Output tensor of shape [B, H/2, W/2, 2C]</returns>
        public Matrix Forward(
            Matrix input,
            Matrix mergeWeight,
            Matrix mergeBias,
            Matrix normWeight,
            Matrix normBias)
        {
            SwinTransformerTools tools = new SwinTransformerTools();
            
            // Convert inputs to PradOp
            this.input = new PradOp(input.ToTensor());
            this.mergeWeight = new PradOp(mergeWeight.ToTensor());
            this.mergeBias = new PradOp(mergeBias.ToTensor());
            this.normWeight = new PradOp(normWeight.ToTensor());
            this.normBias = new PradOp(normBias.ToTensor());

            // Perform patch merging
            var result = tools.PatchMerging(
                this.input,
                this.mergeWeight,
                this.mergeBias,
                this.normWeight,
                this.normBias);
            
            this.resultOp = result.PradOp;
            this.Output = result.Result.ToMatrix();
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            // Perform backward pass using PradOp
            this.resultOp.Back(dLdOutput.ToTensor());

            // Get gradients for all inputs
            var dInput = this.input.SeedGradient.ToMatrix();
            var dMergeWeight = this.mergeWeight.SeedGradient.ToMatrix();
            var dMergeBias = this.mergeBias.SeedGradient.ToMatrix();
            var dNormWeight = this.normWeight.SeedGradient.ToMatrix();
            var dNormBias = this.normBias.SeedGradient.ToMatrix();

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .AddInputGradient(dMergeWeight)
                .AddInputGradient(dMergeBias)
                .AddInputGradient(dNormWeight)
                .AddInputGradient(dNormBias)
                .Build();
        }
    }
}