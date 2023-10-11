using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public interface IDateTimeProvider
    {
        DateTime GetCurrentTime();
    }
}
