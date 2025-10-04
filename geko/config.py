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
    """Kinematic parameter priors (fixed shapes: uniform)"""

    # Asymptotic velocity (km/s) - uniform prior
    Va_min: float = -1000
    Va_max: float = 1000

    # Velocity dispersion (km/s) - uniform prior
    sigma0_min: float = 0
    sigma0_max: float = 500.0

    # Note: r_t (turnover radius) is not configurable - it uses r_eff as max bound

@dataclass
class MCMCSettings:
    """MCMC sampling configuration"""

    num_chains: int = 4
    num_warmup: int = 500
    num_samples: int = 1000
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    step_size: float = 0.1

@dataclass
class FitConfiguration:
    """Configuration for priors and MCMC settings"""

    # Prior configurations
    morphology: MorphologyPriors = None
    kinematics: KinematicPriors = None
    mcmc: MCMCSettings = None
    
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
    
    def validate(self) -> list:
        """
        Validate configuration and return list of issues

        Returns
        -------
        list
            List of validation issues (errors and warnings)
        """
        issues = []

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
        print("Geko Configuration Summary")
        print("=" * 40)

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

        # Nested dataclasses
        if 'morphology' in config_dict:
            config.morphology = MorphologyPriors(**config_dict['morphology'])
        if 'kinematics' in config_dict:
            config.kinematics = KinematicPriors(**config_dict['kinematics'])
        if 'mcmc' in config_dict:
            config.mcmc = MCMCSettings(**config_dict['mcmc'])

        return config
    
    def copy(self) -> 'FitConfiguration':
        """Create a copy of this configuration"""
        config_dict = asdict(self)
        return self._from_dict(config_dict)
    
    @classmethod
    def _from_dict(cls, config_dict: dict) -> 'FitConfiguration':
        """Create configuration from dictionary"""
        config = cls()

        # Nested dataclasses
        if 'morphology' in config_dict:
            config.morphology = MorphologyPriors(**config_dict['morphology'])
        if 'kinematics' in config_dict:
            config.kinematics = KinematicPriors(**config_dict['kinematics'])
        if 'mcmc' in config_dict:
            config.mcmc = MCMCSettings(**config_dict['mcmc'])

        return config


# Main module functions

def get_default_config() -> FitConfiguration:
    """Get a default configuration"""
    return FitConfiguration()


def load_config(filename: str) -> FitConfiguration:
    """Load configuration from file"""
    return FitConfiguration.load(filename)


if __name__ == "__main__":
    # Demo usage
    print("Demo: Creating a configuration")

    # Create config with some custom settings
    config = FitConfiguration(
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