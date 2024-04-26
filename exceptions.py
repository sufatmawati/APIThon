from typing import Dict, List


class Return(Exception):
    """Handles return statements."""

    def __init__(self, value: any, *args: List[any], **kwargs: Dict[any, any]):
        # Call the base class constructor with the parameters it needs
        super().__init__(*args, **kwargs)
        self.value = value


class Break(Exception):
    """Handles break statements."""

    def __init__(self, *args: List[any], **kwargs: Dict[any, any]):
        # Call the base class constructor with the parameters it needs
        super().__init__(*args, **kwargs)


class Continue(Exception):
    """Handles continue statements."""

    def __init__(self, *args: List[any], **kwargs: Dict[any, any]):
        # Call the base class constructor with the parameters it needs
        super().__init__(*args, **kwargs)


class MaximumSizeExceededError(Exception):
    """Thrown when maximum size of memory or any supported type is exceeded."""

    def __init__(self, *args: List[any], **kwargs: Dict[any, any]):
        # Call the base class constructor with the parameters it needs
        super().__init__(*args, **kwargs)


class MaximumOperationExceededError(Exception):
    """Thrown when maximum number of ops is exceeded."""

    def __init__(self, *args: List[any], **kwargs: Dict[any, any]):
        # Call the base class constructor with the parameters it needs
        super().__init__(*args, **kwargs)


class PrivateAccessError(Exception):
    """Thrown when attempting to access private attributes."""

    def __init__(self, *args: List[any], **kwargs: Dict[any, any]):
        # Call the base class constructor with the parameters it needs
        super().__init__(*args, **kwargs)
