﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public enum GradientExpressionType
    {
        None,

        ChainRule,

        CompositePowerRule,

        ProductRule,

        QuotientRule,

        SumRule,

        DifferenceRule,

        Unary
    }
}
