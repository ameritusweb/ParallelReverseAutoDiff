using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.AutoPrad
{
    public class PradExp
    {
        public void BackReplace()
        {

        }

        public void BackAccumulate()
        {

        }

        public static PradExp operator +(PradExp a1, PradExp a2)
        {
            return new PradExp();
        }

        public static PradExp operator -(PradExp a1, PradExp a2)
        {
            return new PradExp();
        }

        public static PradExp operator *(PradExp a1, PradExp a2)
        {
            return new PradExp();
        }

        public static PradExp operator /(PradExp a1, PradExp a2)
        {
            return new PradExp();
        }
    }
}
