namespace ParallelReverseAutoDiff.GravNetExample.GlyphNetwork
{
    using ParallelReverseAutoDiff.RMAD;

    public class GlyphTrainingDynamics
    {
        private static readonly Lazy<GlyphTrainingDynamics> lazy = new Lazy<GlyphTrainingDynamics>(() => new GlyphTrainingDynamics(), true);

        public static GlyphTrainingDynamics Instance { get { return lazy.Value; } }

        public Matrix[] PreviousTargetedSum { get; set; } = new Matrix[2];

        public Matrix[] LastTargetedSum { get; set; } = new Matrix[2];

        public Matrix PreviousRotationAndSum { get; set; }

        public Matrix LastRotationAndSum { get; set; }

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

        public double ActualAngle { get; set; }

        public double ActualAngleTarget0 { get; set; }

        public double ActualAngleTarget1 { get; set; }

        public double TargetAngle { get; set; }

        public double TargetAngle0 { get; set; }

        public double TargetAngle1 { get; set; }

        private GlyphTrainingDynamics()
        {

        }

        public (double, double) CalculateGradientDirection(double x, double y, int targetIndex = -1)
        {
            var actualAngle = Math.Atan2(y, x);

            var targetAngle = TargetAngle;
            if (targetIndex == 0)
            {
                targetAngle = TargetAngle0;
            } else if (targetIndex == 1)
            {
                targetAngle = TargetAngle1;
            }

            var quadrant = GetQuadrant(actualAngle);
            var targetQuadrant = GetQuadrant(targetAngle);
            var oppositeAngle = CalculateOppositeAngle(targetAngle);
            if (targetQuadrant == 1)
            {
                if (quadrant == 1)
                {
                    if (actualAngle < targetAngle)
                    {
                        return (1, -1); // x increase, y decrease
                    }
                    else
                    {
                        return (-1, 1); // x decrease, y increase
                    }
                }
                else if (quadrant == 2)
                {
                    return (-1, -1); // x decrease, y decrease
                }
                else if (quadrant == 3)
                {
                    if (actualAngle < oppositeAngle)
                    {
                        return (1, -1); // x increase, y decrease
                    }
                    else
                    {
                        return (-1, 1); // x decrease, y increase
                    }
                }
                else
                {
                    return (-1, -1); // x decrease, y decrease
                }
            } else if (targetQuadrant == 2)
            {
                if (quadrant == 1)
                {
                    return (1, -1); // x increase, y decrease
                }
                else if (quadrant == 2)
                {
                    if (actualAngle < targetAngle)
                    {
                        return (1, 1); // x increase, y increase
                    }
                    else
                    {
                        return (-1, -1); // x decrease, y decrease
                    }
                }
                else if (quadrant == 3)
                {
                    return (1, -1); // x increase, y decrease
                }
                else
                {
                    if (actualAngle < oppositeAngle)
                    {
                        return (1, 1); // x increase, y increase
                    }
                    else
                    {
                        return (1, -1); // x decrease, y decrease
                    }
                }
            }

            throw new InvalidOperationException("Unsupported target quadrant");
        }

        private int GetQuadrant(double angleInRadians)
        {
            // Normalize the angle to be within 0 to 2pi radians
            double normalizedAngle = angleInRadians % (2 * Math.PI);
            // Adjust if negative to ensure it falls within the 0 to 2pi range
            if (normalizedAngle < 0)
                normalizedAngle += 2 * Math.PI;

            // Determine the quadrant
            if (normalizedAngle >= 0 && normalizedAngle < Math.PI / 2)
                return 1;
            else if (normalizedAngle >= Math.PI / 2 && normalizedAngle < Math.PI)
                return 2;
            else if (normalizedAngle >= Math.PI && normalizedAngle < 3 * Math.PI / 2)
                return 3;
            else // normalizedAngle >= 3 * Math.PI / 2 && normalizedAngle < 2 * Math.PI
                return 4;
        }

        private double CalculateOppositeAngle(double targetAngle)
        {
            // Add π to the target angle to find the opposite angle
            double oppositeAngle = targetAngle + Math.PI;

            // Normalize the opposite angle to be within [-2π, 2π]
            if (oppositeAngle > 2 * Math.PI) oppositeAngle -= 2 * Math.PI;
            else if (oppositeAngle < -2 * Math.PI) oppositeAngle += 2 * Math.PI;

            return oppositeAngle;
        }
    }
}
