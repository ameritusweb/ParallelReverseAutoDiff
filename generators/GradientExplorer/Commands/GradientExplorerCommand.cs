using GradientExplorer;

namespace ToolWindow
{
    [Command(PackageIds.MyExplorerCommand)]
    internal sealed class GradientExplorerCommand : BaseCommand<GradientExplorerCommand>
    {
        protected override Task ExecuteAsync(OleMenuCmdEventArgs e)
        {
            return GradientExplorer.ShowAsync().ContinueWith(x => GradientToolbox.ShowAsync());
        }
    }
}
