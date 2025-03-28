using System;
using System.Collections.Generic;
using System.Text;

namespace ParallelReverseAutoDiff.PRAD.VectorTools
{
    public class VariationConfig
    {
        public float Frequency { get; set; }
        public float Amplitude { get; set; }

        public void Validate()
        {
            if (Frequency <= 0)
                throw new ArgumentException("Frequency must be positive");
            if (Amplitude < 0)
                throw new ArgumentException("Amplitude cannot be negative");
        }
    }
}
