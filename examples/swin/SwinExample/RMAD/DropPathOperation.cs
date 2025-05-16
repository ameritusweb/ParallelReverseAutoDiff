using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.RMAD;

namespace SwinExample.RMAD
{
    public class DropPathOperation : Operation
    {
        private PradOp input;
        private double dropPath;
        private int training;
        private PradOp resultOp;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new DropPathOperation();
        }

        /// <summary>
        /// Performs the forward operation for the drop path function.
        /// </summary>
        /// <param name="input">The input to the classification head operation.</param>
        /// <param name="dropPath">The classifier weight.</param>
        /// <param name="training">The classifier bias.</param>
        /// <returns>The output of the drop path operation.</returns>
        public Matrix Forward(Matrix input, double dropPath, int training)
        {
            SwinTransformerTools tools = new SwinTransformerTools();
            this.input = new PradOp(input.ToTensor());
            this.dropPath = dropPath;
            this.training = training;

            var result = tools.DropPath(this.input, this.dropPath, this.training);
            this.resultOp = result.PradOp;
            this.Output = result.Result.ToMatrix();
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            this.resultOp.Back(dLdOutput.ToTensor());

            var dLdInput = this.input.SeedGradient.ToMatrix();

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }
    }
}
