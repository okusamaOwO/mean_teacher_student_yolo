import torch


class FeatureHookExtractor:
    """Extracts intermediate feature maps from specified layers using forward hooks.
    
    Hooks capture the OUTPUT tensor of each registered layer. The stored tensors
    retain their computational graph so gradients can flow back through them.
    
    Usage:
        extractor = FeatureHookExtractor(model, layer_indices=[17, 20, 23])
        output = model(x)
        feats = extractor.get_features()   # [P3, P4, P5] with grad
        extractor.clear()                  # free memory after use
    """

    def __init__(self, model, layer_indices):
        """
        Args:
            model: YOLOv9 model (may be wrapped in DDP/DataParallel).
            layer_indices (list[int]): Indices into model.model[] to hook.
        """
        self.features = {}
        self.hooks = []
        self.layer_indices = layer_indices

        # Unwrap DDP / DataParallel
        m = model.module if hasattr(model, 'module') else model

        for idx in layer_indices:
            layer = m.model[idx]
            hook = layer.register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, idx):
        def hook_fn(module, input, output):
            # Store output tensor directly — keeps grad graph intact
            self.features[idx] = output
        return hook_fn

    def get_features(self):
        """Returns list of feature tensors in order of layer_indices."""
        feats = []
        for idx in self.layer_indices:
            assert idx in self.features, \
                f"Feature for layer {idx} not captured. Did you run model.forward()?"
            feats.append(self.features[idx])
        return feats

    def clear(self):
        """Clear stored features to free memory (call after backward)."""
        self.features.clear()

    def remove_hooks(self):
        """Remove all hooks permanently."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.features.clear()
