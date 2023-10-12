using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Helpers
{
    public struct PublishOptions
    {
        public bool AllowNoSubscribers { get; set; }

        public static PublishOptions Default => new PublishOptions { AllowNoSubscribers = false };
    }
}
