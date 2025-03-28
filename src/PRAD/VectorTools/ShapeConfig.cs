using System;
using System.Collections.Generic;
using System.Text;

namespace ParallelReverseAutoDiff.PRAD.VectorTools
{
    public class ShapeConfig
    {
        public int NumPoints { get; set; }
        public List<Segment> Segments { get; set; }

        public void Validate()
        {
            if (NumPoints <= 0)
                throw new ArgumentException("NumPoints must be positive");
            if (Segments == null || Segments.Count == 0)
                throw new ArgumentException("At least one segment must be specified");

            foreach (var segment in Segments)
            {
                segment.Validate();
            }
        }
    }
}
