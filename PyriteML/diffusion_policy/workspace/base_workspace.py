# filepath: /home/robotlab/ACP/adaptive_compliance_policy/PyriteML/diffusion_policy/workspace/base_workspace.py
from typing import Optional
import os
import pathlib
import hydra
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading


class BaseWorkspace:
    """
    Base class for all training/evaluation workspaces.
    
    Provides standardized checkpoint management, configuration handling,
    and serialization for ML experiments. Subclasses implement specific
    training loops and model architectures.
    """
    
    # Subclasses can override these to control what gets saved/loaded
    include_keys = tuple()  # Non-PyTorch objects to explicitly save (e.g., counters, custom state)
    exclude_keys = tuple()  # PyTorch objects to skip saving (e.g., temporary modules)

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None):
        """
        Initialize workspace with configuration.
        
        Args:
            cfg: OmegaConf configuration tree (from YAML + command line overrides)
            output_dir: Optional override for output directory (defaults to Hydra's output dir)
        """
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None  # For non-blocking checkpoint saving

    @property
    def output_dir(self):
        """Get output directory, using Hydra's default if not specified."""
        output_dir = self._output_dir
        if output_dir is None:
            # Use Hydra's automatically generated output directory
            # (organized by timestamp, config name, etc.)
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def run(self):
        """
        Main execution entry point for the workspace.
        
        Subclasses override this to implement training loops, evaluation, etc.
        Any non-serializable resources (datasets, loggers) should be created here
        as local variables to avoid checkpoint bloat.
        """
        pass

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=True):
        """
        Save training checkpoint with model states and training progress.
        
        Automatically detects PyTorch modules (model, optimizer, scheduler) and saves
        their state_dict(). Also saves specified non-PyTorch objects (counters, etc.).
        
        Args:
            path: Custom save path (defaults to output_dir/checkpoints/{tag}.ckpt)
            tag: Checkpoint name/tag (e.g., 'latest', 'best', 'epoch_100')
            exclude_keys: PyTorch modules to skip saving
            include_keys: Non-PyTorch objects to save (counters, custom state)
            use_thread: Save in background thread to avoid blocking training
            
        Returns:
            str: Absolute path where checkpoint was saved
        """
        # Default path: output_dir/checkpoints/{tag}.ckpt
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
            
        # Use class defaults if not specified
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        # Ensure checkpoint directory exists
        path.parent.mkdir(parents=False, exist_ok=True)
        
        # Checkpoint payload structure
        payload = {
            'cfg': self.cfg,           # Complete configuration for reproducibility
            'state_dicts': dict(),     # PyTorch module states (model, optimizer, etc.)
            'pickles': dict()          # Other serializable objects (counters, custom state)
        } 

        # Automatically detect and save all PyTorch modules
        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # This is a PyTorch module (model, optimizer, scheduler, etc.)
                if key not in exclude_keys:
                    if use_thread:
                        # Copy to CPU to avoid GPU memory issues during background saving
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
                        
            elif key in include_keys:
                # Explicitly included non-PyTorch objects (step counters, etc.)
                payload['pickles'][key] = dill.dumps(value)
                
        # Save checkpoint (optionally in background thread)
        if use_thread:
            # Non-blocking save - training continues while checkpoint saves
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            # Blocking save - training waits for checkpoint to complete
            torch.save(payload, path.open('wb'), pickle_module=dill)
            
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        """Get path to checkpoint file for given tag."""
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        """
        Load checkpoint payload into workspace.
        
        Restores PyTorch module states and pickled objects from checkpoint.
        Called by load_checkpoint() and create_from_checkpoint().
        
        Args:
            payload: Checkpoint dictionary with 'state_dicts' and 'pickles'
            exclude_keys: Skip loading these PyTorch modules
            include_keys: Load these pickled objects (defaults to all)
            **kwargs: Additional arguments for load_state_dict()
        """
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        # Restore PyTorch module states (model weights, optimizer momentum, etc.)
        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
                
        # Restore pickled objects (step counters, custom state, etc.)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        """
        Load checkpoint and restore workspace state.
        
        Args:
            path: Custom checkpoint path (defaults to output_dir/checkpoints/{tag}.ckpt)
            tag: Checkpoint tag to load if path not specified
            exclude_keys: Skip loading these PyTorch modules  
            include_keys: Load these pickled objects
            **kwargs: Additional arguments for load_state_dict()
            
        Returns:
            dict: The loaded checkpoint payload
        """
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
            
        # Load checkpoint file
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        
        # Restore workspace state from payload
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        """
        Create workspace instance directly from checkpoint file.
        
        Useful for inference/evaluation without recreating training setup.
        
        Args:
            path: Path to checkpoint file
            exclude_keys: Skip loading these PyTorch modules
            include_keys: Load these pickled objects
            **kwargs: Additional arguments for load_state_dict()
            
        Returns:
            BaseWorkspace: New workspace instance loaded from checkpoint
        """
        # Load checkpoint payload
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        
        # Create workspace instance with saved configuration
        instance = cls(payload['cfg'])
        
        # Load saved state into new instance
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Save complete workspace state for quick debugging/research.
        
        Unlike checkpoints, snapshots save the ENTIRE workspace object,
        including datasets, loggers, etc. Fast to save/load but fragile
        to code changes.
        
        Use for short-term debugging, not long-term storage.
        
        Args:
            tag: Snapshot name/tag
            
        Returns:
            str: Absolute path where snapshot was saved
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        
        # Save entire workspace object (everything!)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        """
        Load workspace from snapshot (complete object serialization).
        
        Assumes code hasn't changed since snapshot was saved.
        
        Args:
            path: Path to snapshot file
            
        Returns:
            BaseWorkspace: Loaded workspace instance
        """
        return torch.load(open(path, 'rb'), pickle_module=dill)


def _copy_to_cpu(x):
    """
    Recursively copy PyTorch tensors to CPU for background checkpoint saving.
    
    This prevents GPU memory conflicts when saving checkpoints in a background
    thread while training continues on GPU.
    
    Args:
        x: PyTorch tensor, dict, list, or other object
        
    Returns:
        Object with all tensors moved to CPU
    """
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)