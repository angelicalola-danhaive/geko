"""
Rotation curve components and composite rotation curve for geko.

Adding a new model: subclass RotationCurveComponent, define _DEFAULT_PARAMETERS,
v_sq(), and velocity_sign(), then add to COMPONENT_REGISTRY.
"""

__all__ = [
    'RotationCurveComponent', 'CompositeRotationCurve',
    'ArctanComponent', 'COMPONENT_REGISTRY',
]

from abc import abstractmethod
import jax.numpy as jnp

from .param_spec import BaseModel, ParameterSpec


class RotationCurveComponent(BaseModel):
    """Abstract base for a single rotation curve contribution.

    Each component contributes a v²(r) term; the composite sums them.
    Components with NEEDS_PHYSICAL_SCALE=True receive kpc_per_px at construction.
    """
    NEEDS_PHYSICAL_SCALE: bool = False

    @abstractmethod
    def v_sq(self, r_px: jnp.ndarray, all_params: dict) -> jnp.ndarray:
        """Return v²(r) contribution in (km/s)²."""
        ...

    @abstractmethod
    def velocity_sign(self, all_params: dict) -> jnp.ndarray:
        """Return the sign (±1) of the rotation direction from this component's parameters."""
        ...

    def sample(self, morph_params: dict) -> dict:
        return self._sample(context=morph_params)


class CompositeRotationCurve:
    """Container that sums v² contributions from all components.

    Not a BaseModel — sigma0 lives in SHARED_KINEMATIC_SPEC and is accessible
    via all_params when rotation_curve() is called (for future asymmetric drift
    correction).
    """

    def __init__(self, components: list):
        self.components = components

    @property
    def parameters(self) -> list:
        return [s for c in self.components for s in c.parameters]

    def sample(self, morph_params: dict) -> dict:
        params = {}
        for c in self.components:
            params.update(c.sample(morph_params))
        return params

    def rotation_curve(self, r_px: jnp.ndarray, all_params: dict) -> jnp.ndarray:
        # TODO: asymmetric drift correction using all_params['sigma0']
        v2 = sum(c.v_sq(r_px, all_params) for c in self.components)
        sign = jnp.array(1.0)
        for c in self.components:
            sign = sign * c.velocity_sign(all_params)
        return sign * jnp.sqrt(jnp.clip(v2, 0.0))


# ---------------------------------------------------------------------------
# Concrete components
# ---------------------------------------------------------------------------

class ArctanComponent(RotationCurveComponent):
    """Phenomenological arctangent rotation curve: v(r) = Va * (2/π) * arctan(r/r_t)."""
    NEEDS_PHYSICAL_SCALE = False
    _DEFAULT_PARAMETERS = [
        ParameterSpec('Va',  r'$v_a$ [km/s]', r'$v_a$', 'Uniform',
                      prior_min=-1000.0, prior_max=1000.0),
        ParameterSpec('r_t', r'$r_t$ [px]',   r'$r_t$', 'Uniform',
                      prior_min=0.0, prior_max=None),  # prior_max=None → r_eff from context
    ]

    def v_sq(self, r_px: jnp.ndarray, all_params: dict) -> jnp.ndarray:
        v = all_params['Va'] * (2.0 / jnp.pi) * jnp.arctan(r_px / all_params['r_t'])
        return v ** 2

    def velocity_sign(self, all_params: dict) -> jnp.ndarray:
        return jnp.sign(all_params['Va'])


COMPONENT_REGISTRY = {
    'Arctan': ArctanComponent,
}
