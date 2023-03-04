__doc__ = """ File for training callbacks.

The main base class is `Callback`, and other callbacks are implemented as subclasses.
See the `Callback` class for details."""

from abc import ABC
from typing import Dict

from torch.utils.tensorboard import SummaryWriter


class Callback(ABC):
    """
    Base class used for other callbacks.
    Default implementation has no-ops for all methods.
    Subclasses are not required to override any given method,
    because any given subclass is likely to only override a subset
    and different subclasses may override different subsets.

    Attributes:
        trainer ("trainers.Trainer"): Trainer to which this callback is attached.

    """

    def __init__(self):
        self.trainer = None

    def set_trainer(self, trainer: "trainers.Trainer"):
        if self.trainer is not None:
            raise RuntimeError("Callback already has a trainer set")
        self.trainer = trainer

    def on_train_start(self):
        """Called before main training loop starts"""
        pass

    def on_train_end(self):
        """Called after main training loop is finished"""
        pass

    def on_epoch_start(self, epoch_num: int):
        """
        Called before each training epoch
        Args:
            epoch_num: Training epoch index
        """
        pass

    def on_epoch_end(self, epoch_num: int, epoch_results: Dict):
        """
        Called after each epoch is finished.
        Args:
            epoch_num: Training epoch index
            epoch_results: Dict of results from the epoch
        """
        pass


class DecayParameter(Callback):
    """
    Callback used to decay a training parameter.
    Decay is geometric:

        value(episode+1) = decay*value(episode)

    This update value is used until `value` reaches `min_value`

    Attributes:
        name (str): Name of the parameter. This is used as the key
            in `self.trainer.train_kwargs`.
        init (float): Initial value of the parameter.
        decay (float): Decay rate
        min_value (float): Lowest possible value. The decay effectively stops
            once the parameter reaches this value.
    """

    def __init__(self, name: str, init: float, decay: float, min_value: float = 0.0):
        super().__init__()
        self.name = name
        self.init = init
        if init < 0:
            raise ValueError("Initial value must be non-negative")
        self.decay = decay
        self.min_value = min_value

    def on_train_start(self):
        """Initializes parameter value"""
        self.trainer.train_kwargs[self.name] = self.init

    def on_epoch_end(self, epoch_num: int, epoch_results: Dict):
        """Updates parameter value"""
        old_value = self.trainer.train_kwargs.get(self.name, 0.0)
        new_value = self.decay * old_value
        new_value = max(new_value, self.min_value)
        self.trainer.train_kwargs[self.name] = new_value


class PrintResults(Callback):
    """Basic callback which prints interesting quantities to the console.

    Mainly here to help demonstrate what a Callback can do, but potentially
    also useful for debugging."""

    def __init__(self, print_interval=10):
        super().__init__()
        self.print_interval = print_interval

    def on_epoch_end(self, epoch_num: int, epoch_results: Dict):
        """Prints results from the epoch"""
        print_interval = self.print_interval
        train_kwargs = self.trainer.train_kwargs
        if epoch_num > 0 and epoch_num % print_interval == 0:
            total_reward = epoch_results["total_reward"]
            mean_loss = epoch_results["mean_loss"]
            print_str = f"Epoch {epoch_num}"
            print_str += f" Reward: {total_reward:0.4}"
            print_str += f" Mean Loss: {mean_loss:0.4}"
            if "epsilon" in train_kwargs:
                epsilon = train_kwargs["epsilon"]
                print_str += f" epsilon: {epsilon:0.4}"
            print(print_str)


class TensorboardCallback(Callback):
    """
    Callback which logs training results to Tensorboard
    """

    def __init__(self, log_dir: str, **kwargs):
        """
        Args:
            log_dir: Directory for storing log files
            **kwargs: Other keywords passed to torch.utils.tensorboard.SummaryWriter
        """
        super().__init__()
        self.log_dir = log_dir
        self.kwargs = kwargs
        self.writer = None

    def on_train_start(self):
        """Open SummaryWriter"""
        self.writer = SummaryWriter(log_dir=self.log_dir, **self.kwargs)

    def on_epoch_end(self, epoch_num: int, epoch_results: Dict):
        """Add scalar results from the training epoch"""
        for key, val in epoch_results.items():
            self.writer.add_scalar(key, val, epoch_num)

    def on_train_end(self):
        """Close SummaryWriter.

        This will write any buffered data and close any open files.
        """
        self.writer.close()
