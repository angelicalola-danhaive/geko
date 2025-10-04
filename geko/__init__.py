from .fitting import *
from .grism import *
from .models import *
from .plotting import *
from .postprocess import *
from .preprocess import *
from .utils import *
from .config import *

# High-level convenience functions
def fit_galaxy(data_path, config, **kwargs):
    """
    High-level function to fit a galaxy with given configuration.
    
    Parameters
    ----------
    data_path : str
        Path to the galaxy data file
    config : FitConfiguration
        Configuration object with priors and settings
    **kwargs
        Additional keyword arguments passed to run_inference
    
    Returns
    -------
    arviz.InferenceData
        MCMC inference results
    """
    from .fitting import Fit_Numpyro
    from . import preprocess as pre
    
    # Validate config
    issues = config.validate()
    errors = [issue for issue in issues if issue.startswith("ERROR")]
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  {error}")
        raise ValueError("Configuration validation failed")
    
    # Show warnings
    warnings = [issue for issue in issues if issue.startswith("WARNING")]
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  {warning}")
    
    # Print configuration summary
    config.print_summary()
    
    # TODO: Implement data loading and preprocessing
    # This would replace the current run_full_preprocessing call
    # For now, raise NotImplementedError with instructions
    raise NotImplementedError(
        "fit_galaxy is not yet fully implemented. "
        "For now, use the existing workflow:\n"
        "1. Create your config: config = geko.config.FitConfiguration(...)\n"
        "2. Use existing preprocessing and Fit_Numpyro class\n"
        "3. Pass config to Fit_Numpyro(..., config=config)"
    )


__all__ = ["Fit_Numpyro", "Grism", "KinModels", 'plot_disk_summary', 'plot_pp_cornerplot', 'process_results', 'run_full_preprocessing',
           'oversample', 'resample', 'scale_distribution', 'find_best_sample', 'compute_gal_props',
           'load_psf', 'compute_inclination', 'compute_axis_ratio', 'add_v_re', 'sersic_profile',
           'compute_adaptive_sersic_profile', 'flux_to_Ie', 'Ie_to_flux', 'fit_galaxy',
           'FitConfiguration', 'MorphologyPriors', 'KinematicPriors', 'MCMCSettings',
           'get_default_config', 'load_config']