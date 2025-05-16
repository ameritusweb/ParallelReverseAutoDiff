using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.RMAD;

namespace SwinExample.RMAD
{
    /// <summary>
    /// Operation that implements window attention mechanism in Swin Transformer.
    /// </summary>
    public class WindowAttentionOperation : Operation
    {
        private PradOp input;
        private PradOp qkvWeight;
        private PradOp qkvBias;
        private PradOp projWeight;
        private PradOp projBias;
        private PradOp relativePosTable;
        private PradOp resultOp;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new WindowAttentionOperation();
        }

        /// <summary>
        /// Performs the window attention forward operation.
        /// </summary>
        /// <param name="input">Input tensor of shape [num_windows*B, N, C]</param>
        /// <param name="qkvWeight">Weight matrix for QKV projection</param>
        /// <param name="qkvBias">Bias for QKV projection</param>
        /// <param name="projWeight">Weight matrix for output projection</param>
        /// <param name="projBias">Bias for output projection</param>
        /// <param name="relativePosTable">Relative position bias table</param>
        /// <param name="options">Configuration tensor [numHeads, windowSize, attentionDropout, projectionDropout, training]</param>
        /// <returns>Output tensor of shape [num_windows*B, N, C]</returns>
        public Matrix Forward(
            Matrix input,
            Matrix qkvWeight,
            Matrix qkvBias,
            Matrix projWeight,
            Matrix projBias,
            Matrix relativePosTable,
            Matrix options)
        {
            SwinTransformerTools tools = new SwinTransformerTools();
            
            // Convert inputs to PradOp
            this.input = new PradOp(input.ToTensor());
            this.qkvWeight = new PradOp(qkvWeight.ToTensor());
            this.qkvBias = new PradOp(qkvBias.ToTensor());
            this.projWeight = new PradOp(projWeight.ToTensor());
            this.projBias = new PradOp(projBias.ToTensor());
            this.relativePosTable = new PradOp(relativePosTable.ToTensor());
            var optionsPradOp = new PradOp(options.ToTensor());

            // Compute window attention
            var result = tools.WindowAttention(
                this.input,
                this.qkvWeight,
                this.qkvBias,
                this.projWeight,
                this.projBias,
                this.relativePosTable,
                optionsPradOp);

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
            var dQkvWeight = this.qkvWeight.SeedGradient.ToMatrix();
            var dQkvBias = this.qkvBias.SeedGradient.ToMatrix();
            var dProjWeight = this.projWeight.SeedGradient.ToMatrix();
            var dProjBias = this.projBias.SeedGradient.ToMatrix();
            var dRelativePosTable = this.relativePosTable.SeedGradient.ToMatrix();

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .AddInputGradient(dQkvWeight)
                .AddInputGradient(dQkvBias)
                .AddInputGradient(dProjWeight)
                .AddInputGradient(dProjBias)
                .AddInputGradient(dRelativePosTable)
                .Build();
        }
    }
}