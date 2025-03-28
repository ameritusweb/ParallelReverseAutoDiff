using System;
using System.Collections.Generic;
using System.Text;

namespace ParallelReverseAutoDiff.PRAD.VectorTools
{
    public class ShapeConfigs
    {
        public Dictionary<string, ShapeConfig> Shapes { get; set; }

        public void Validate()
        {
            if (Shapes == null || Shapes.Count == 0)
                throw new ArgumentException("At least one shape must be specified");

            foreach (var shape in Shapes.Values)
            {
                shape.Validate();
            }
        }
    }
}
