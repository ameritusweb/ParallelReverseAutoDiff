using System;
using System.Collections.Generic;
using System.Text;

namespace ParallelReverseAutoDiff.PRAD.VectorTools
{
    public class Segment
    {
        public float StartAngle { get; set; }
        public float EndAngle { get; set; }
        public RadiusConfig OuterRadius { get; set; }
        public RadiusConfig InnerRadius { get; set; }

        public void Validate()
        {
            if (StartAngle >= EndAngle)
                throw new ArgumentException("StartAngle must be less than EndAngle");

            OuterRadius?.Validate();
            InnerRadius?.Validate();

            if (OuterRadius == null || InnerRadius == null)
                throw new ArgumentException("Both OuterRadius and InnerRadius must be specified");

            // Check that inner radius is always less than outer radius, even with variations
            float maxInnerRadius = InnerRadius.Base + (InnerRadius.Variation?.Amplitude ?? 0);
            float minOuterRadius = OuterRadius.Base - (OuterRadius.Variation?.Amplitude ?? 0);

            if (maxInnerRadius >= minOuterRadius)
                throw new ArgumentException("InnerRadius must always be less than OuterRadius");
        }
    }
}
