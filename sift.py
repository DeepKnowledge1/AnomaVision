import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math


class SIFTDemo:
    def __init__(self, image_path):
        """
        Initialize SIFT demonstration with an image
        """
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original_image is None:
            # Create a synthetic image if file not found
            self.original_image = self.create_synthetic_image()

        # SIFT Parameters
        self.num_octaves = 4
        self.num_scales = 5  # Number of blur levels per octave
        self.sigma = 1.6  # Base sigma for Gaussian blur
        self.k = 2 ** (1 / 3)  # Scale factor between adjacent scales

        print(f"Original image size: {self.original_image.shape}")
        print(f"Number of octaves: {self.num_octaves}")
        print(f"Scales per octave: {self.num_scales}")

    def create_synthetic_image(self):
        """Create a synthetic test image with various features"""
        img = np.zeros((400, 400), dtype=np.uint8)

        # Add some geometric shapes
        cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
        cv2.circle(img, (300, 100), 50, 128, -1)
        cv2.rectangle(img, (200, 250), (350, 350), 200, 2)

        # Add some noise for texture
        noise = np.random.normal(0, 10, img.shape)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

        print("Using synthetic test image (original image not found)")
        return img

    def build_octaves(self):
        """
        Step 1: Build different octaves (image resolutions)
        Each octave is half the size of the previous one
        """
        print("\n🏔️ STEP 1: Building Octaves")
        print("=" * 50)

        octaves = []
        current_image = self.original_image.copy()

        for i in range(self.num_octaves):
            octaves.append(current_image.copy())
            height, width = current_image.shape

            print(f"Octave {i+1}: {width} × {height} pixels")

            # For next octave, downsample by factor of 2
            if i < self.num_octaves - 1:
                current_image = cv2.pyrDown(current_image)

        self.octaves = octaves
        return octaves

    def build_gaussian_pyramid(self):
        """
        Step 2: For each octave, create multiple Gaussian blurred versions
        """
        print("\n🔍 STEP 2: Building Gaussian Scale-Space")
        print("=" * 50)

        self.gaussian_pyramid = []

        for octave_idx, octave_image in enumerate(self.octaves):
            print(f"\nProcessing Octave {octave_idx + 1}:")

            octave_scales = []

            for scale_idx in range(self.num_scales):
                # Calculate sigma for this scale
                sigma = self.sigma * (self.k**scale_idx)

                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(octave_image, (0, 0), sigma)
                octave_scales.append(blurred)

                print(f"  Scale {scale_idx + 1}: σ = {sigma:.2f}")

            self.gaussian_pyramid.append(octave_scales)

        return self.gaussian_pyramid

    def build_dog_pyramid(self):
        """
        Step 3: Create Difference of Gaussians (DoG) by subtracting adjacent scales
        """
        print("\n⚡ STEP 3: Building DoG (Difference of Gaussians)")
        print("=" * 50)

        self.dog_pyramid = []

        for octave_idx, octave_scales in enumerate(self.gaussian_pyramid):
            print(f"\nProcessing DoG for Octave {octave_idx + 1}:")

            octave_dogs = []

            # Create DoG by subtracting adjacent Gaussian images
            for i in range(len(octave_scales) - 1):
                dog = octave_scales[i + 1].astype(np.int16) - octave_scales[i].astype(
                    np.int16
                )
                octave_dogs.append(dog)

                print(f"  DoG {i + 1}: Scale {i + 2} - Scale {i + 1}")

            self.dog_pyramid.append(octave_dogs)

        return self.dog_pyramid

    def find_keypoints(self):
        """
        Step 4: Find keypoints by looking for extrema in DoG space
        """
        print("\n📍 STEP 4: Finding Keypoints (Extrema Detection)")
        print("=" * 50)

        all_keypoints = []

        for octave_idx, octave_dogs in enumerate(self.dog_pyramid):
            print(f"\nSearching keypoints in Octave {octave_idx + 1}:")
            octave_keypoints = []

            # Need at least 3 DoG images to find extrema
            if len(octave_dogs) < 3:
                continue

            # Check middle DoG images (not first or last)
            for dog_idx in range(1, len(octave_dogs) - 1):
                keypoints = self.detect_extrema(
                    octave_dogs[dog_idx - 1],  # Previous scale
                    octave_dogs[dog_idx],  # Current scale
                    octave_dogs[dog_idx + 1],  # Next scale
                )

                # Scale keypoints back to original image coordinates
                scale_factor = 2**octave_idx
                scaled_keypoints = [
                    (int(x * scale_factor), int(y * scale_factor), octave_idx, dog_idx)
                    for x, y in keypoints
                ]

                octave_keypoints.extend(scaled_keypoints)
                print(f"  DoG {dog_idx}: Found {len(keypoints)} keypoints")

            all_keypoints.extend(octave_keypoints)
            print(f"  Total keypoints in octave: {len(octave_keypoints)}")

        self.keypoints = all_keypoints
        print(f"\n🎯 Total keypoints found: {len(all_keypoints)}")
        return all_keypoints

    def detect_extrema(self, dog_prev, dog_curr, dog_next, threshold=10):
        """
        Detect local extrema (keypoints) in DoG space
        """
        keypoints = []
        height, width = dog_curr.shape

        # Check each pixel (except borders)
        for y in range(1, height - 1):
            for x in range(1, width - 1):

                center_val = dog_curr[y, x]

                # Skip if value is too small
                if abs(center_val) < threshold:
                    continue

                # Check if it's a local maximum or minimum
                is_extrema = True

                # Check 3x3x3 neighborhood
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        for dz, dog_img in enumerate([dog_prev, dog_curr, dog_next]):
                            if dy == 0 and dx == 0 and dz == 1:  # Skip center pixel
                                continue

                            neighbor_val = dog_img[y + dy, x + dx]

                            # Check for maximum
                            if center_val > 0 and neighbor_val >= center_val:
                                is_extrema = False
                                break
                            # Check for minimum
                            elif center_val < 0 and neighbor_val <= center_val:
                                is_extrema = False
                                break

                        if not is_extrema:
                            break
                    if not is_extrema:
                        break

                if is_extrema:
                    keypoints.append((x, y))

        return keypoints

    def visualize_results(self):
        """
        Create comprehensive visualization of all steps
        """
        print("\n🎨 Creating Visualizations...")

        # Create a large figure
        fig = plt.figure(figsize=(20, 16))

        # 1. Show original image and octaves
        plt.subplot(4, 6, 1)
        plt.imshow(self.original_image, cmap="gray")
        plt.title("Original Image", fontsize=10)
        plt.axis("off")

        for i, octave in enumerate(self.octaves[:4]):
            plt.subplot(4, 6, i + 2)
            plt.imshow(octave, cmap="gray")
            plt.title(f"Octave {i+1}\n{octave.shape[1]}×{octave.shape[0]}", fontsize=9)
            plt.axis("off")

        # 2. Show Gaussian pyramid for first octave
        row_start = 7
        for i, gaussian_img in enumerate(self.gaussian_pyramid[0][:5]):
            plt.subplot(4, 6, row_start + i)
            plt.imshow(gaussian_img, cmap="gray")
            sigma = self.sigma * (self.k**i)
            plt.title(f"Gaussian σ={sigma:.2f}", fontsize=9)
            plt.axis("off")

        # 3. Show DoG pyramid for first octave
        row_start = 13
        for i, dog_img in enumerate(self.dog_pyramid[0][:4]):
            plt.subplot(4, 6, row_start + i)
            plt.imshow(dog_img, cmap="RdBu_r", vmin=-50, vmax=50)
            plt.title(f"DoG {i+1}", fontsize=9)
            plt.axis("off")

        # 4. Show keypoints on original image
        plt.subplot(4, 6, 6)
        plt.imshow(self.original_image, cmap="gray")

        # Plot keypoints with different colors for different octaves
        colors = ["red", "blue", "green", "yellow"]
        for x, y, octave, scale in self.keypoints:
            if (
                0 <= x < self.original_image.shape[1]
                and 0 <= y < self.original_image.shape[0]
            ):
                plt.plot(
                    x,
                    y,
                    "o",
                    color=colors[octave % len(colors)],
                    markersize=3,
                    alpha=0.7,
                )

        plt.title(f"Keypoints Found\n{len(self.keypoints)} points", fontsize=10)
        plt.axis("off")

        plt.tight_layout()
        plt.suptitle("SIFT Scale-Space Construction Process", fontsize=16, y=0.98)
        plt.show()

    def run_complete_demo(self):
        """
        Run the complete SIFT scale-space construction demo
        """
        print("🚀 Starting SIFT Scale-Space Construction Demo")
        print("=" * 60)

        # Step 1: Build octaves
        self.build_octaves()

        # Step 2: Build Gaussian pyramid
        self.build_gaussian_pyramid()

        # Step 3: Build DoG pyramid
        self.build_dog_pyramid()

        # Step 4: Find keypoints
        self.find_keypoints()

        # Step 5: Visualize everything
        self.visualize_results()

        print("\n✅ Demo completed successfully!")
        print(f"📊 Summary:")
        print(f"   • Created {self.num_octaves} octaves")
        print(f"   • {self.num_scales} Gaussian scales per octave")
        print(f"   • {self.num_scales-1} DoG images per octave")
        print(f"   • Found {len(self.keypoints)} total keypoints")


# Example usage
if __name__ == "__main__":
    # Initialize demo (will create synthetic image if no file provided)
    demo = SIFTDemo("your_image.jpg")  # Replace with your image path

    # Run the complete demonstration
    demo.run_complete_demo()

    # You can also run individual steps:
    print("\n" + "=" * 60)
    print("🔍 Individual Step Demonstrations:")
    print("=" * 60)

    # Just show octaves
    octaves = demo.build_octaves()

    # Show individual octave sizes
    print(f"\n📐 Octave Details:")
    for i, octave in enumerate(octaves):
        print(f"Octave {i+1}: {octave.shape} - {octave.nbytes} bytes")

    # Show Gaussian parameters
    print(f"\n🔍 Gaussian Blur Parameters:")
    for scale_idx in range(demo.num_scales):
        sigma = demo.sigma * (demo.k**scale_idx)
        print(f"Scale {scale_idx + 1}: σ = {sigma:.3f}")
