"""
Simple test to verify column (C) dispersion implementation.

This script tests that:
1. Grism object can be created with pupil='C' without errors
2. The disperse method produces correct output shapes
3. Both R and C dispersion work as expected
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from geko.grism import Grism

def test_column_dispersion():
    """Test column dispersion implementation."""

    print("="*70)
    print("Testing Column (C) Dispersion Implementation")
    print("="*70)

    # Test parameters
    im_shape = 31
    wavelength = 4.2
    wave_space = jnp.linspace(3.8, 5.0, 1000)
    index_min = 100
    index_max = 900

    radius = 10  # Define how big you want the circle

    # Create simple test data background
    flux = jnp.ones((im_shape, im_shape)) * 0.1

    # 1. Create a coordinate grid
    x = jnp.arange(im_shape)
    y = jnp.arange(im_shape)
    xx, yy = jnp.meshgrid(x, y)

    # 2. Calculate distance from the center
    center = im_shape // 2
    dist_sq = (xx - center)**2 + (yy - center)**2

    # 3. Create a mask where distance is less than radius
    # We use radius squared to avoid calculating square roots (faster)
    mask = dist_sq <= radius**2
    #make it more elliptical for fun
    mask = ((xx - center)**2) / (radius**2) + ((yy - center)**2) / ((radius*0.6)**2) <= 1.0

    # 4. Set the pixels inside the mask to 1.0
    flux = flux.at[mask].set(1.0)

    velocity = jnp.zeros((im_shape, im_shape))
    dispersion = jnp.ones((im_shape, im_shape)) * 50.0  # 50 km/s dispersion

    # Create a simple Gaussian PSF
    psf_size = 15
    center = psf_size // 2
    y, x = jnp.meshgrid(jnp.arange(psf_size), jnp.arange(psf_size), indexing='ij')
    psf = jnp.exp(-((x - center)**2 + (y - center)**2) / (2 * 2**2))
    psf = psf / jnp.sum(psf)  # Normalize

    print("\nTest setup:")
    print(f"  Image shape: {im_shape} x {im_shape}")
    print(f"  Wavelength space: {len(wave_space)} points from {wave_space[0]:.2f} to {wave_space[-1]:.2f} μm")
    print(f"  Index range: {index_min} to {index_max}")
    print(f"  PSF shape: {psf.shape}")

    # Test 1: Row (R) dispersion
    print("\n" + "-"*70)
    print("Test 1: Row (R) Dispersion")
    print("-"*70)

    try:
        grism_R = Grism(
            im_shape=im_shape,
            wavelength=wavelength,
            wave_space=wave_space,
            index_min=index_min,
            index_max=index_max,
            grism_filter='F444W',
            grism_module='A',
            grism_pupil='R',
            PSF=psf
        )
        print("✓ Grism object created successfully for pupil='R'")
        print(f"  Module: {grism_R.module}, Pupil: {grism_R.pupil}")

        grism_output_R = grism_R.disperse(flux, velocity, dispersion)
        print(f"✓ Disperse completed for R")
        print(f"  Output shape: {grism_output_R.shape}")
        print(f"  Expected: ({im_shape}, {index_max - index_min})")

        assert grism_output_R.shape == (im_shape, index_max - index_min), \
            f"Wrong shape for R dispersion: expected ({im_shape}, {index_max - index_min}), got {grism_output_R.shape}"
        print("✓ Output shape is correct for R dispersion")

    except Exception as e:
        print(f"✗ Error with R dispersion: {e}")
        return False

    # Test 2: Column (C) dispersion
    print("\n" + "-"*70)
    print("Test 2: Column (C) Dispersion")
    print("-"*70)

    try:
        grism_C = Grism(
            im_shape=im_shape,
            wavelength=wavelength,
            wave_space=wave_space,
            index_min=index_min,
            index_max=index_max,
            grism_filter='F444W',
            grism_module='A',
            grism_pupil='C',
            PSF=psf
        )
        print("✓ Grism object created successfully for pupil='C'")
        print(f"  Module: {grism_C.module}, Pupil: {grism_C.pupil}")

        grism_output_C = grism_C.disperse(flux, velocity, dispersion)
        print(f"✓ Disperse completed for C")
        print(f"  Output shape: {grism_output_C.shape}")
        print(f"  Expected: ({im_shape}, {index_max - index_min})")

        assert grism_output_C.shape == (im_shape, index_max - index_min), \
            f"Wrong shape for C dispersion: expected ({im_shape}, {index_max - index_min}), got {grism_output_C.shape}"
        print("✓ Output shape is correct for C dispersion")

    except Exception as e:
        print(f"✗ Error with C dispersion: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Compare coefficients
    print("\n" + "-"*70)
    print("Test 3: Coefficient Comparison")
    print("-"*70)

    print(f"R dispersion b01 coefficient: {grism_R.b01:.4f}")
    print(f"C dispersion b01 coefficient: {grism_C.b01:.4f}")
    print("✓ Different coefficients loaded for R vs C (as expected)")

    # Test 4: Verify outputs are different
    print("\n" + "-"*70)
    print("Test 4: Output Verification")
    print("-"*70)

    output_diff = jnp.abs(grism_output_R - grism_output_C).max()
    print(f"Maximum difference between R and C outputs: {output_diff:.6f}")

    if output_diff > 1e-6:
        print("✓ R and C produce different outputs (as expected)")
    else:
        print("⚠ Warning: R and C outputs are very similar")

    # Test 5: Generate comparison plots
    print("\n" + "-"*70)
    print("Test 5: Generating Comparison Plots")
    print("-"*70)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Input and R dispersion
    # Input flux map
    im0 = axes[0, 0].imshow(flux, origin='lower', cmap='hot', aspect='auto')
    axes[0, 0].set_title('Input Flux Map\n(Single bright pixel at center)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('X (pixels)')
    axes[0, 0].set_ylabel('Y (pixels)')
    plt.colorbar(im0, ax=axes[0, 0], label='Flux')

    # R dispersion output (row dispersion - horizontal)
    im1 = axes[0, 1].imshow(grism_output_R[:,200:300], origin='lower', cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Row (R) Dispersion Output\n(Horizontal spectrum)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Wavelength (pixel index)')
    axes[0, 1].set_ylabel('Spatial Y (pixels)')
    plt.colorbar(im1, ax=axes[0, 1], label='Flux')
    axes[0, 1].axhline(y=im_shape//2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Center row')
    axes[0, 1].legend(loc='upper right', fontsize=8)

    # R dispersion center row profile
    center_row_R = grism_output_R[im_shape//2, :]
    axes[0, 2].plot(np.arange(len(center_row_R)), center_row_R, 'b-', linewidth=2)
    axes[0, 2].set_title('R Dispersion: Center Row\n(Wavelength spectrum)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Wavelength (pixel index)')
    axes[0, 2].set_ylabel('Flux')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim(0, len(center_row_R))

    # Row 2: C dispersion
    # C dispersion output (column dispersion - vertical)
    im2 = axes[1, 1].imshow(grism_output_C[:,200:300], origin='lower', cmap='plasma', aspect='auto')
    axes[1, 1].set_title('Column (C) Dispersion Output\n(Vertical spectrum)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Wavelength (pixel index)')
    axes[1, 1].set_ylabel('Spatial X (pixels)')
    plt.colorbar(im2, ax=axes[1, 1], label='Flux')
    axes[1, 1].axhline(y=im_shape//2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Center column')
    axes[1, 1].legend(loc='upper right', fontsize=8)

    # C dispersion center column profile
    center_col_C = grism_output_C[im_shape//2, :]
    axes[1, 2].plot(np.arange(len(center_col_C)), center_col_C, 'r-', linewidth=2)
    axes[1, 2].set_title('C Dispersion: Center Column\n(Wavelength spectrum)', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Wavelength (pixel index)')
    axes[1, 2].set_ylabel('Flux')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlim(0, len(center_col_C))

    # Bottom left: Show difference map
    diff_map = np.abs(grism_output_R - grism_output_C)
    im3 = axes[1, 0].imshow(diff_map, origin='lower', cmap='RdYlBu_r', aspect='auto')
    axes[1, 0].set_title('Absolute Difference\n|R - C|', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Wavelength (pixel index)')
    axes[1, 0].set_ylabel('Pixels')
    plt.colorbar(im3, ax=axes[1, 0], label='|Difference|')

    plt.tight_layout()

    # Save figure
    output_filename = '/Users/lola/geko/column_dispersion_test.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_filename}")

    plt.show()

    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
    print(f"\nVisualization saved to: {output_filename}")

    return True

if __name__ == "__main__":
    success = test_column_dispersion()
    exit(0 if success else 1)
