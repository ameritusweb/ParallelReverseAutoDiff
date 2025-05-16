using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.RMAD;

namespace SwinExample.RMAD
{
    /// <summary>
    /// Operation that implements cyclic shift of features, used in shifted window attention.
    /// </summary>
    public class CyclicShiftOperation : Operation
    {
        private PradOp input;
        private PradOp resultOp;
        private readonly int shift;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new CyclicShiftOperation();
        }

        /// <summary>
        /// Initializes a new instance of the CyclicShiftOperation class.
        /// </summary>
        public CyclicShiftOperation(int shift = -1)
        {
            this.shift = shift;
        }

        /// <summary>
        /// Performs the forward operation for cyclic shift.
        /// </summary>
        /// <param name="input">The input tensor of shape [B, H, W, C]</param>
        /// <param name="shift">Amount to shift features</param>
        /// <returns>The cyclically shifted tensor</returns>
        public Matrix Forward(Matrix input, int shift)
        {
            SwinTransformerTools tools = new SwinTransformerTools();
            this.input = new PradOp(input.ToTensor());
            
            var result = tools.CyclicShift(this.input, shift);
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