using System;
using System.Collections.Generic;
using System.Text;

namespace ParallelReverseAutoDiff.PRAD.VectorTools
{
    public class RadiusConfig
    {
        public float Base { get; set; }
        public VariationConfig? Variation { get; set; }

        public void Validate()
        {
            if (Base <= 0)
                throw new ArgumentException("Base radius must be positive");
            Variation?.Validate();
        }
    }
}
