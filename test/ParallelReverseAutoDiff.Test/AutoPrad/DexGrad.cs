using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.AutoPrad
{
    public class DexGrad
    {
        public Tensor this[Tensor t]
        {
            get
            {
                return t;
            }
        }
    }
}
