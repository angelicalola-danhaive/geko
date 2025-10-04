"""
Configuration system for geko fitting parameters.
Provides programmatic setup for scientific parameters with validation.
"""

import numpy as np
import yaml
from dataclasses import dataclass, asdict, fields
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

@dataclass
class MorphologyPriors:
    """Morphological parameter priors (fixed shapes: uniform/truncated normal)"""
    
    # Position angle (degrees) - uniform prior
    PA_min: float = 0.0
    PA_max: float = 180.0
    
    # Inclination (degrees) - uniform prior
    inc_min: float = 30.0
    inc_max: float = 80.0
    
    # Effective radius (pixels) - truncated normal
    r_eff_mean: float = 3.0
    r_eff_std: float = 1.0
    r_eff_min: float = 0.5
    r_eff_max: float = 10.0
    
    # Sersic index - truncated normal
    n_mean: float = 1.0
    n_std: float = 0.5
    n_min: float = 0.5
    n_max: float = 4.0
    
    # Central coordinates (pixels from center) - normal
    xc_mean: float = 0.0
    xc_std: float = 2.0
    yc_mean: float = 0.0
    yc_std: float = 2.0
    
    # Amplitude - log-normal
    amplitude_mean: float = 100.0
    amplitude_std: float = 50.0
    amplitude_min: float = 1.0
    amplitude_max: float = 1000.0

@dataclass 
class KinematicPriors:
    """Kinematic parameter priors (fixed shapes: uniform/truncated normal)"""
    
    # Asymptotic velocity (km/s) - uniform prior
    Va_min: float = 50.0
    Va_max: float = 500.0
    
    # Turnover radius (pixels) - truncated normal
    r_t_mean: float = 2.0
    r_t_std: float = 1.0
    r_t_min: float = 0.1
    r_t_max: float = 8.0
    
    # Velocity dispersion (km/s) - uniform prior
    sigma0_min: float = 20.0
    sigma0_max: float = 200.0

@dataclass
class MCMCSettings:
    """MCMC sampling configuration"""
    
    num_chains: int = 4
    num_warmup: int = 500
    num_samples: int = 1000
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    step_size: Optional[float] = None

@dataclass
class ComputationSettings:
    """Computational settings for JAX/Numpyro"""
    
    platform: str = "auto"  # "auto", "cpu", "gpu"
    host_device_count: Optional[int] = None  # Auto-detect if None
    enable_x64: bool = True
    validation: bool = True

@dataclass
class FitConfiguration:
    """Complete fit configuration"""
    
    # Required observational parameters
    redshift: Optional[float] = None
    line: str = "H_alpha"
    filter: str = "F444W"
    
    # Model settings
    parametric: bool = False
    
    # Prior configurations
    morphology: MorphologyPriors = None
    kinematics: KinematicPriors = None
    mcmc: MCMCSettings = None
    computation: ComputationSettings = None
    
    def __post_init__(self):
        # Store defaults for comparison (to track which params were explicitly modified)
        self._default_morphology = MorphologyPriors()
        self._default_kinematics = KinematicPriors()

        if self.morphology is None:
            self.morphology = MorphologyPriors()
        if self.kinematics is None:
            self.kinematics = KinematicPriors()
        if self.mcmc is None:
            self.mcmc = MCMCSettings()
        if self.computation is None:
            self.computation = ComputationSettings()
    
    def validate(self) -> list:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check required parameters
        if self.redshift is None:
            issues.append("ERROR: redshift must be specified")
        elif not 0 < self.redshift < 15:
            issues.append(f"WARNING: redshift {self.redshift} is outside typical range (0-15)")
        
        # Validate morphology priors
        if self.morphology.PA_min >= self.morphology.PA_max:
            issues.append("ERROR: PA_min must be less than PA_max")
        
        if self.morphology.inc_min >= self.morphology.inc_max:
            issues.append("ERROR: inc_min must be less than inc_max")
        
        if self.morphology.inc_min < 20:
            issues.append("WARNING: Very low inclinations (<20°) may be difficult to fit")
        
        if self.morphology.r_eff_min >= self.morphology.r_eff_max:
            issues.append("ERROR: r_eff_min must be less than r_eff_max")
        
        # Validate kinematic priors
        if self.kinematics.Va_min >= self.kinematics.Va_max:
            issues.append("ERROR: Va_min must be less than Va_max")
        
        if self.kinematics.sigma0_min >= self.kinematics.sigma0_max:
            issues.append("ERROR: sigma0_min must be less than sigma0_max")
        
        if self.kinematics.r_t_min >= self.kinematics.r_t_max:
            issues.append("ERROR: r_t_min must be less than r_t_max")
        
        # Validate MCMC settings
        if self.mcmc.num_chains < 1:
            issues.append("ERROR: num_chains must be at least 1")
        
        if self.mcmc.target_accept_prob <= 0 or self.mcmc.target_accept_prob >= 1:
            issues.append("ERROR: target_accept_prob must be between 0 and 1")

        return issues

    def get_modified_params(self):
        """
        Return dictionaries of only the parameters that differ from defaults.

        This allows selective override of priors - only explicitly modified
        parameters will override PySersic priors or other defaults.

        Returns
        -------
        dict
            Dictionary with 'morphology' and 'kinematics' keys, each containing
            only the parameters that were explicitly modified from defaults.
        """
        modified = {'morphology': {}, 'kinematics': {}}

        # Check morphology parameters
        morph_dict = asdict(self.morphology)
        default_morph_dict = asdict(self._default_morphology)
        for key, value in morph_dict.items():
            if value != default_morph_dict[key]:
                modified['morphology'][key] = value

        # Check kinematic parameters
        kin_dict = asdict(self.kinematics)
        default_kin_dict = asdict(self._default_kinematics)
        for key, value in kin_dict.items():
            if value != default_kin_dict[key]:
                modified['kinematics'][key] = value

        return modified

    def print_summary(self):
        """Print a summary of the configuration, highlighting non-default values"""
        print("Geko Fit Configuration Summary")
        print("=" * 40)
        
        # Basic settings
        print("\nObservational Parameters:")
        print(f"  redshift: {self.redshift}")
        print(f"  line: {self.line}")
        print(f"  filter: {self.filter}")
        print(f"  parametric: {self.parametric}")
        
        # Show validation issues
        issues = self.validate()
        if issues:
            print(f"\nValidation Issues ({len(issues)}):")
            for issue in issues:
                print(f"  {issue}")
        
        # Morphology priors
        print(f"\nMorphology Priors:")
        self._print_section_summary(self.morphology, MorphologyPriors())
        
        # Kinematic priors
        print(f"\nKinematic Priors:")
        self._print_section_summary(self.kinematics, KinematicPriors())
        
        # MCMC settings
        print(f"\nMCMC Settings:")
        self._print_section_summary(self.mcmc, MCMCSettings())
        
        # Computation settings
        print(f"\nComputation Settings:")
        self._print_section_summary(self.computation, ComputationSettings())
        
        # Runtime estimate
        total_samples = self.mcmc.num_chains * (self.mcmc.num_warmup + self.mcmc.num_samples)
        est_minutes = max(1, int(total_samples * 0.1 / 60))
        print(f"\nEstimated Runtime: ~{est_minutes} minutes")
        print("=" * 40)
    
    def _print_section_summary(self, current_obj, default_obj):
        """Print summary of a configuration section, highlighting changes"""
        current_dict = asdict(current_obj)
        default_dict = asdict(default_obj)
        
        for key, current_value in current_dict.items():
            default_value = default_dict[key]
            if current_value != default_value:
                print(f"  {key}: {current_value} (default: {default_value})")
            else:
                print(f"  {key}: {current_value}")
    
    def save(self, filename: str, output_dir: str = None):
        """
        Save configuration to YAML file with timestamp
        
        Parameters
        ----------
        filename : str
            Name of the configuration file (timestamp will be added automatically)
        output_dir : str, optional
            Directory to save the configuration. If None, saves in current directory.
        """
        import os
        import datetime
        
        config_dict = asdict(self)
        
        # Add metadata with timestamp
        now = datetime.datetime.now()
        timestamp_readable = now.strftime("%Y-%m-%d %H:%M:%S")  # For metadata (human readable)
        
        config_dict['_metadata'] = {
            'geko_version': '1.0.0',  # TODO: get from package
            'created_by': 'geko.config',
            'created_at': timestamp_readable
        }
        
        # Use filename as provided (no timestamp added)
        if not (filename.endswith('.yaml') or filename.endswith('.yml')):
            filename = f"{filename}.yaml"
        
        # Construct full path
        if output_dir is not None:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            full_path = os.path.join(output_dir, filename)
        else:
            full_path = filename
        
        with open(full_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"Configuration saved to {full_path}")
    
    @classmethod
    def load(cls, filename: str) -> 'FitConfiguration':
        """Load configuration from YAML file"""
        with open(filename, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Remove metadata if present
        config_dict.pop('_metadata', None)
        
        # Reconstruct nested dataclasses
        config = cls()
        
        # Basic parameters
        for key in ['redshift', 'line', 'filter', 'parametric']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        # Nested dataclasses
        if 'morphology' in config_dict:
            config.morphology = MorphologyPriors(**config_dict['morphology'])
        if 'kinematics' in config_dict:
            config.kinematics = KinematicPriors(**config_dict['kinematics'])
        if 'mcmc' in config_dict:
            config.mcmc = MCMCSettings(**config_dict['mcmc'])
        if 'computation' in config_dict:
            config.computation = ComputationSettings(**config_dict['computation'])
        
        return config
    
    def copy(self) -> 'FitConfiguration':
        """Create a copy of this configuration"""
        config_dict = asdict(self)
        return self._from_dict(config_dict)
    
    @classmethod
    def _from_dict(cls, config_dict: dict) -> 'FitConfiguration':
        """Create configuration from dictionary"""
        config = cls()
        
        # Basic parameters
        for key in ['redshift', 'line', 'filter', 'parametric']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        # Nested dataclasses
        if 'morphology' in config_dict:
            config.morphology = MorphologyPriors(**config_dict['morphology'])
        if 'kinematics' in config_dict:
            config.kinematics = KinematicPriors(**config_dict['kinematics'])
        if 'mcmc' in config_dict:
            config.mcmc = MCMCSettings(**config_dict['mcmc'])
        if 'computation' in config_dict:
            config.computation = ComputationSettings(**config_dict['computation'])
        
        return config


# Convenience functions and templates

def create_high_redshift_config(redshift: float) -> FitConfiguration:
    """Create configuration template for high redshift galaxies (z>2)"""
    config = FitConfiguration(
        redshift=redshift,
        line="H_alpha",
        filter="F444W",
        parametric=False
    )
    
    # Adjust priors for high-z
    config.kinematics.Va_min = 100
    config.kinematics.Va_max = 600
    config.kinematics.sigma0_min = 30
    config.kinematics.sigma0_max = 150
    config.morphology.inc_min = 40  # Avoid face-on for better kinematics
    
    return config


def create_quick_test_config(redshift: float) -> FitConfiguration:
    """Create configuration for quick testing (faster MCMC)"""
    config = FitConfiguration(
        redshift=redshift,
        line="H_alpha",
        filter="F444W", 
        parametric=True  # Faster
    )
    
    # Faster MCMC settings
    config.mcmc.num_chains = 2
    config.mcmc.num_warmup = 200
    config.mcmc.num_samples = 300
    
    return config


def create_conservative_config(redshift: float) -> FitConfiguration:
    """Create configuration with conservative/broad priors"""
    config = FitConfiguration(
        redshift=redshift,
        line="H_alpha",
        filter="F444W",
        parametric=False
    )
    
    # Broad priors
    config.morphology.inc_min = 20
    config.morphology.inc_max = 85
    config.kinematics.Va_min = 20
    config.kinematics.Va_max = 800
    config.kinematics.sigma0_min = 10
    config.kinematics.sigma0_max = 300
    
    return config


# Main module functions

def get_default_config() -> FitConfiguration:
    """Get a default configuration (user must set redshift)"""
    return FitConfiguration()


def load_config(filename: str) -> FitConfiguration:
    """Load configuration from file"""
    return FitConfiguration.load(filename)


if __name__ == "__main__":
    # Demo usage
    print("Demo: Creating a configuration")
    
    # Create config with some custom settings
    config = FitConfiguration(
        redshift=3.2,
        morphology=MorphologyPriors(
            PA_min=10,
            PA_max=170,
            inc_min=40,
            inc_max=70
        ),
        kinematics=KinematicPriors(
            Va_max=400
        ),
        mcmc=MCMCSettings(
            num_samples=800
        )
    )
    
    # Show summary
    config.print_summary()
    
    # Save example
    config.save("example_config.yaml")