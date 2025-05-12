using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.AutoPrad
{
    public class AutoPrad
    {
        public AutoGrad Gradients { get; set; }

        public PradExp DoOp(string operation, Tensor t1, Tensor t2)
        {
            return new PradExp();
        }

        public PradExp DoOp(string operation, PradExp e1, PradExp e2)
        {
            return new PradExp();
        }

        public PradExp DoOp(string operation, PradExp e1, Tensor t2)
        {
            return new PradExp();
        }

        public PradExp DoOp(Expression<Func<PradExp>> exp)
        {
            return new PradExp();
        }

        public Tensor Forward()
        {
            return null;
        }
    }
}
