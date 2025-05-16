using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.RMAD;

namespace SwinExample.RMAD
{
    public class ClassificationHeadOperation : Operation
    {
        private PradOp input;
        private PradOp classifierWeight;
        private PradOp classifierBias;
        private PradOp resultOp;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ClassificationHeadOperation();
        }

        /// <summary>
        /// Performs the forward operation for the classifcation head function.
        /// </summary>
        /// <param name="input">The input to the classification head operation.</param>
        /// <param name="classifierWeight">The classifier weight.</param>
        /// <param name="classifierBias">The classifier bias.</param>
        /// <param name="numClasses">The number of classes.</param>
        /// <returns>The output of the classification head operation.</returns>
        public Matrix Forward(Matrix input, Matrix classifierWeight, Matrix classifierBias, int numClasses)
        {
            SwinTransformerTools tools = new SwinTransformerTools();
            this.input = new PradOp(input.ToTensor());
            this.classifierWeight = new PradOp(classifierWeight.ToTensor());
            this.classifierBias = new PradOp(classifierBias.ToTensor());

            var result = tools.ClassificationHead(this.input, this.classifierWeight, this.classifierBias, numClasses);
            this.resultOp = result.PradOp;
            this.Output = result.Result.ToMatrix();
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            this.resultOp.Back(dLdOutput.ToTensor());

            var dLdInput = this.input.SeedGradient.ToMatrix();
            var dWeight = this.classifierWeight.SeedGradient.ToMatrix();
            var dBias = this.classifierBias.SeedGradient.ToMatrix();

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .AddInputGradient(dWeight)
                .AddInputGradient(dBias)
                .Build();
        }
    }
}
