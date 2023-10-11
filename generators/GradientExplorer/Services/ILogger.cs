using GradientExplorer.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Services
{
    public interface ILogger
    {
        void Log(string message, SeverityType severity);

        void SetMinSeverity(SeverityType newMinSeverity);
    }
}
