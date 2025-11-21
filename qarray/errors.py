class QArrayError(Exception):
    """Base exception for QArray-related errors."""

    pass


class QArrayConfigError(QArrayError):
    """Invalid configuration of QArray or coupler topology."""

    pass


class QArrayIndexError(QArrayError):
    """Invalid index or label for QArray."""

    pass


class QArrayValueError(QArrayError):
    """Invalid value (e.g., illegal label format)."""

    pass
