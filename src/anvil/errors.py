from __future__ import annotations


class AnvilError(Exception):
    """Base class for all Anvil-specific errors."""


class PinNotFound(AnvilError):
    """Raised when a named pin cannot be located on disk."""


class ManifestVersionMismatch(AnvilError):
    """Raised when a manifest's schema version is incompatible with this Anvil."""


class CameraError(AnvilError):
    """Raised when a camera driver fails to open, read, or close a device."""


class RobotError(AnvilError):
    """Raised when a robot driver fails to read state or apply configuration."""
