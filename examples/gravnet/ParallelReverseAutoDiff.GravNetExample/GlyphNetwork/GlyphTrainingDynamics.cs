namespace ParallelReverseAutoDiff.GravNetExample.GlyphNetwork
{
    using ParallelReverseAutoDiff.RMAD;

    public class GlyphTrainingDynamics
    {
        private static readonly Lazy<GlyphTrainingDynamics> lazy = new Lazy<GlyphTrainingDynamics>(() => new GlyphTrainingDynamics(), true);

        public static GlyphTrainingDynamics Instance { get { return lazy.Value; } }

        public Matrix[] PreviousTargetedSum { get; set; } = new Matrix[2];

        public Matrix[] LastTargetedSum { get; set; } = new Matrix[2];

        public double PreviousAngleLoss { get; set; }

        public double PreviousEuclideanLoss { get; set; }

        public double PreviousMagnitudeLoss { get; set; }

        public double AngleLoss { get; set; }

        public double EuclideanLoss { get; set; }

        public double MagnitudeLoss { get; set; }

        public double GradAngleLossX { get; set; }

        public double GradAngleLossY { get; set; }

        public double GradEuclideanLossX { get; set; }

        public double GradEuclideanLossY { get; set; }

        public double GradMagnitudeLossX { get; set; }

        public double GradMagnitudeLossY { get; set; }

        private GlyphTrainingDynamics()
        {

        }
    }
}
