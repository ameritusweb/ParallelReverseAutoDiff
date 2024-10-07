using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV;

public class CIELABGradientProcessor
{
    public static (Image<Gray, float>[] Magnitude, Image<Gray, float>[] Angle) ProcessLABImageWithPolarVectors(Image<Lab, float> labImage)
    {
        // Create images to store the magnitudes and angles (L, a, b)
        var magnitudes = new Image<Gray, float>[3];
        var angles = new Image<Gray, float>[3];

        // Initialize output images with the same size as the input
        for (int i = 0; i < 3; i++)
        {
            magnitudes[i] = new Image<Gray, float>(labImage.Width, labImage.Height);
            angles[i] = new Image<Gray, float>(labImage.Width, labImage.Height);
        }

        // Loop over the image with a sliding 3x3 window
        for (int x = 1; x < labImage.Width - 1; x++)
        {
            for (int y = 1; y < labImage.Height - 1; y++)
            {
                // Extract the 3x3 window for the current position
                var labWindow = labImage.GetSubRect(new System.Drawing.Rectangle(x - 1, y - 1, 3, 3));

                // Calculate the polar gradients for the window
                var gradients = CIELABGradientCalculator.CalculateLABGradientsPolar(labWindow);

                // Store the results in the respective output images
                magnitudes[0].Data[y, x, 0] = gradients["L"].Magnitude;
                angles[0].Data[y, x, 0] = gradients["L"].AngleRadians;

                magnitudes[1].Data[y, x, 0] = gradients["a"].Magnitude;
                angles[1].Data[y, x, 0] = gradients["a"].AngleRadians;

                magnitudes[2].Data[y, x, 0] = gradients["b"].Magnitude;
                angles[2].Data[y, x, 0] = gradients["b"].AngleRadians;
            }
        }

        // Return the three magnitude and angle images (L, a, b)
        return (magnitudes, angles);
    }

    public static Image<Lab, float> ConvertBgrToLabWithPadding(Image<Bgr, float> bgrImage)
    {
        // Manually create a padded image with a 1-pixel border
        var paddedBgrImage = new Image<Bgr, float>(bgrImage.Width + 2, bgrImage.Height + 2);

        // Copy the original image into the center of the new image
        bgrImage.CopyTo(paddedBgrImage.GetSubRect(new System.Drawing.Rectangle(1, 1, bgrImage.Width, bgrImage.Height)));

        // Replicate the border by copying edge pixels to the border
        for (int x = 0; x < bgrImage.Width; x++)
        {
            // Top and bottom rows
            paddedBgrImage.Data[0, x + 1, 0] = bgrImage.Data[0, x, 0];
            paddedBgrImage.Data[paddedBgrImage.Height - 1, x + 1, 0] = bgrImage.Data[bgrImage.Height - 1, x, 0];

            // Left and right columns
            paddedBgrImage.Data[x + 1, 0, 0] = bgrImage.Data[x, 0, 0];
            paddedBgrImage.Data[x + 1, paddedBgrImage.Width - 1, 0] = bgrImage.Data[x, bgrImage.Width - 1, 0];
        }

        // Convert to LAB color space
        return paddedBgrImage.Convert<Lab, float>();
    }
}