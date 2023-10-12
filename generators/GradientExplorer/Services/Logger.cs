using GradientExplorer.Helpers;
using Microsoft.VisualStudio.Shell.Interop;

namespace GradientExplorer.Services
{
    public class Logger : ILogger
    {
        private readonly IPaneCreator paneCreator;
        private readonly IDateTimeProvider dateTimeProvider;
        private readonly IEnvironmentProvider environmentProvider;
        private IVsOutputWindowPane pane;
        private SeverityType minSeverity;

        public Logger(IPaneCreator paneCreator, IDateTimeProvider dateTimeProvider, IEnvironmentProvider environmentProvider, SeverityType minSeverity = SeverityType.Information)
        {
            this.paneCreator = paneCreator;
            this.dateTimeProvider = dateTimeProvider;
            this.environmentProvider = environmentProvider;
            this.minSeverity = minSeverity;
            pane = paneCreator.CreatePane(Guid.NewGuid(), "Gradient Explorer", true, true);
        }

        public void SetMinSeverity(SeverityType newMinSeverity)
        {
            this.minSeverity = newMinSeverity;
        }

        public void Log(string message, SeverityType severity)
        {
            ThreadHelper.ThrowIfNotOnUIThread();

            if (severity < minSeverity)
            {
                return;
            }

            string formattedMessage = FormatMessage(message, severity);
            try
            {
                var res = pane.OutputStringThreadSafe(formattedMessage + environmentProvider.GetNewLine());
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Logging failed unexpectedly for message [{formattedMessage}]: {ex.Message}");
            }
        }

        private string FormatMessage(string message, SeverityType severity)
        {
            return $"[{dateTimeProvider.GetCurrentTime().ToString("HH:mm:ss")}] {severity.ToString().ToUpperInvariant()}: {message}";
        }
    }
}
