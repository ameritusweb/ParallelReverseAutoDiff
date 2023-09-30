using GradientExplorer;

namespace ToolWindow
{
    [Command(PackageIds.MyCommand)]
    internal sealed class GradientExplorerCommand : BaseCommand<GradientExplorerCommand>
    {
        protected override Task ExecuteAsync(OleMenuCmdEventArgs e)
        {
            return GradientExplorer.ShowAsync();
        }
    }
}
