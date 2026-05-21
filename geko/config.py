"""
Configuration system for geko fitting parameters.
"""

__all__ = ["FitConfiguration", "MCMCSettings"]

from dataclasses import dataclass, field, asdict
import yaml


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
    """Configuration for geko fitting.

    Parameters
    ----------
    mcmc : MCMCSettings
        MCMC sampler settings.
    morphology_model : str
        Name of morphology model from MORPH_REGISTRY (default: 'Sersic').
    rotation_components : list of str
        Ordered list of rotation curve component names from COMPONENT_REGISTRY
        (default: ['Arctan']).
    morph_prior_overrides : dict
        Override morphology ParameterSpec priors. Keys follow the pattern
        {param_name}_{mu|std|min|max}, e.g. 'r_eff_mu', 'n_std'.
    rot_prior_overrides : dict
        Override rotation component ParameterSpec priors (same key convention).
    geom_prior_overrides : dict
        Override shared kinematic ParameterSpec priors (PA, i, sigma0, v0, etc.).
    fixed_params : dict
        Pin or link parameters. Values can be:
        - float: pin the parameter to that value
        - str: link to another sampled parameter by name

    Examples
    --------
    YAML equivalent::

        morphology_model: Sersic
        rotation_components: [Arctan]
        morph_prior_overrides:
          r_eff_max: 20.0
        rot_prior_overrides:
          Va_max: 400.0
        geom_prior_overrides:
          sigma0_max: 300.0
        fixed_params:
          i: 60.0
    """

    mcmc: MCMCSettings = field(default_factory=MCMCSettings)
    morphology_model: str = 'Sersic'
    rotation_components: list = field(default_factory=lambda: ['Arctan'])
    morph_prior_overrides: dict = field(default_factory=dict)
    rot_prior_overrides: dict = field(default_factory=dict)
    geom_prior_overrides: dict = field(default_factory=dict)
    fixed_params: dict = field(default_factory=dict)

    def print_summary(self):
        """Print a summary of the configuration."""
        print("Geko Configuration Summary")
        print("=" * 40)
        print(f"  morphology_model:    {self.morphology_model}")
        print(f"  rotation_components: {self.rotation_components}")
        if self.morph_prior_overrides:
            print(f"  morph_prior_overrides: {self.morph_prior_overrides}")
        if self.rot_prior_overrides:
            print(f"  rot_prior_overrides:   {self.rot_prior_overrides}")
        if self.geom_prior_overrides:
            print(f"  geom_prior_overrides:  {self.geom_prior_overrides}")
        if self.fixed_params:
            print(f"  fixed_params:          {self.fixed_params}")
        print(f"\nMCMC Settings:")
        for f_name in ('num_chains', 'num_warmup', 'num_samples',
                       'target_accept_prob', 'max_tree_depth', 'step_size'):
            print(f"  {f_name}: {getattr(self.mcmc, f_name)}")

    def save(self, filename: str, output_dir: str = None):
        """Save configuration to YAML file."""
        import os, datetime

        config_dict = {
            'morphology_model': self.morphology_model,
            'rotation_components': self.rotation_components,
            'morph_prior_overrides': self.morph_prior_overrides,
            'rot_prior_overrides': self.rot_prior_overrides,
            'geom_prior_overrides': self.geom_prior_overrides,
            'fixed_params': self.fixed_params,
            'mcmc': asdict(self.mcmc),
            '_metadata': {
                'geko_version': '2.0.0',
                'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        if not (filename.endswith('.yaml') or filename.endswith('.yml')):
            filename = f"{filename}.yaml"
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, filename)

        with open(filename, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        print(f"Configuration saved to {filename}")

    @classmethod
    def load(cls, filename: str) -> 'FitConfiguration':
        """Load configuration from YAML file."""
        with open(filename, 'r') as f:
            d = yaml.safe_load(f)

        d.pop('_metadata', None)
        mcmc_dict = d.pop('mcmc', {})

        cfg = cls(
            morphology_model=d.get('morphology_model', 'Sersic'),
            rotation_components=d.get('rotation_components', ['Arctan']),
            morph_prior_overrides=d.get('morph_prior_overrides', {}),
            rot_prior_overrides=d.get('rot_prior_overrides', {}),
            geom_prior_overrides=d.get('geom_prior_overrides', {}),
            fixed_params=d.get('fixed_params', {}),
        )
        if mcmc_dict:
            cfg.mcmc = MCMCSettings(**mcmc_dict)
        return cfg
