__all__ = [ "PublishingState"]

class _State:
    """
    Simple object to store the state of a setting
    """
    def __init__(self) -> None:
        self._setting_set = False
    
    def turn_on(self):
        self._setting_set = True
    
    def turn_off(self):
        self._setting_set = False
    
    def is_set(self):
        return self._setting_set


class PublishingState(_State):
    def __init__(self) -> None:
        """
        Simple object to store the state of a plotting-style setting
        """
        super().__init__()
