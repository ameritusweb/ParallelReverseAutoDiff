using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.RMAD;

namespace SwinExample.RMAD
{
    /// <summary>
    /// Operation that implements the MLP (FFN) block in Swin Transformer.
    /// </summary>
    public class MLPOperation : Operation
    {
        private PradOp input;
        private PradOp fc1Weight;
        private PradOp fc1Bias;
        private PradOp fc2Weight;
        private PradOp fc2Bias;
        private PradOp resultOp;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MLPOperation();
        }

        /// <summary>
        /// Performs the forward operation for the MLP block.
        /// </summary>
        /// <param name="input">Input tensor of shape [B, L, C]</param>
        /// <param name="fc1Weight">First fully connected layer weights</param>
        /// <param name="fc1Bias">First fully connected layer bias</param>
        /// <param name="fc2Weight">Second fully connected layer weights</param>
        /// <param name="fc2Bias">Second fully connected layer bias</param>
        /// <param name="options">Options tensor containing [dropoutRate, training]</param>
        /// <returns>Output tensor of shape [B, L, C]</returns>
        public Matrix Forward(
            Matrix input,
            Matrix fc1Weight,
            Matrix fc1Bias,
            Matrix fc2Weight,
            Matrix fc2Bias,
            Matrix options)
        {
            SwinTransformerTools tools = new SwinTransformerTools();
            
            // Convert inputs to PradOp
            this.input = new PradOp(input.ToTensor());
            this.fc1Weight = new PradOp(fc1Weight.ToTensor());
            this.fc1Bias = new PradOp(fc1Bias.ToTensor());
            this.fc2Weight = new PradOp(fc2Weight.ToTensor());
            this.fc2Bias = new PradOp(fc2Bias.ToTensor());
            var optionsPradOp = new PradOp(options.ToTensor());

            // Compute MLP
            var result = tools.MLP(
                this.input,
                this.fc1Weight,
                this.fc1Bias,
                this.fc2Weight,
                this.fc2Bias,
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
            var dFc1Weight = this.fc1Weight.SeedGradient.ToMatrix();
            var dFc1Bias = this.fc1Bias.SeedGradient.ToMatrix();
            var dFc2Weight = this.fc2Weight.SeedGradient.ToMatrix();
            var dFc2Bias = this.fc2Bias.SeedGradient.ToMatrix();

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .AddInputGradient(dFc1Weight)
                .AddInputGradient(dFc1Bias)
                .AddInputGradient(dFc2Weight)
                .AddInputGradient(dFc2Bias)
                .Build();
        }
    }
}