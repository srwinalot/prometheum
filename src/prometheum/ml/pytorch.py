"""
PyTorch Adapter for Prometheum ML Framework.

This module provides an adapter for integrating PyTorch models with
Prometheum data processing pipelines. It handles model creation, training,
evaluation, and serialization.
"""

import os
import json
import importlib
import inspect
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast, Callable

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    import torchmetrics
except ImportError:
    raise ImportError(
        "PyTorch is required for this module. "
        "Install it with: pip install torch torchmetrics"
    )

from prometheum.core.base import DataFrameContainer, DataTransformer
from prometheum.core.exceptions import ProcessingError
from prometheum.ml.base import (
    FrameworkType,
    MLAdapter,
    ModelCreationError,
    ModelInfo,
    ModelType,
    TrainingError,
    PredictionError,
    EvaluationError,
    SerializationError,
)


def _get_torch_model_type(model: Any) -> ModelType:
    # ... (unchanged helper as you provided)
    if not isinstance(model, nn.Module):
        return ModelType.CUSTOM
    name = model.__class__.__name__.lower()
    if any(term in name for term in ('classifier','classification')):
        return ModelType.CLASSIFIER
    if any(term in name for term in ('regressor','regression')):
        return ModelType.REGRESSOR
    if any(term in name for term in ('autoencoder','encoder','embedding','transformer')):
        return ModelType.TRANSFORMER
    if any(term in name for term in ('gan','generator')):
        return ModelType.NEURAL_NETWORK
    if hasattr(model, '_model_type') and isinstance(model._model_type, ModelType):
        return model._model_type
    # Fallback on final layer analysis...
    children = list(model.children()) if hasattr(model, 'children') else []
    if children:
        last = children[-1]
        if isinstance(last, nn.Linear) and last.out_features == 1:
            return ModelType.REGRESSOR
        if isinstance(last, nn.Linear) and last.out_features > 1:
            if any(isinstance(m, nn.Softmax) for m in model.modules()):
                return ModelType.CLASSIFIER
            if hasattr(model, 'loss_fn') and 'crossentropy' in str(model.loss_fn).lower():
                return ModelType.CLASSIFIER
    return ModelType.NEURAL_NETWORK


def _extract_torch_model_params(model: Any) -> Dict[str, Any]:
    # ... (unchanged extraction logic)
    params: Dict[str, Any] = {}
    if not isinstance(model, nn.Module):
        return params
    try:
        params["architecture"] = model.__class__.__name__
        mods, count = [], 0
        for idx,(name,module) in enumerate(model.named_modules()):
            if not name: continue
            info = {"idx":idx,"name":name,"type":module.__class__.__name__}
            if isinstance(module, nn.Linear):
                info.update({"in_features":module.in_features,"out_features":module.out_features,"bias":module.bias is not None})
            elif isinstance(module, nn.Conv2d):
                info.update({
                    "in_channels":module.in_channels,
                    "out_channels":module.out_channels,
                    "kernel_size":module.kernel_size,
                    "stride":module.stride,
                    "padding":module.padding
                })
            mods.append(info)
        params["modules"] = mods
        params["num_modules"] = len(mods)
        params["num_parameters"] = sum(p.numel() for p in model.parameters())
        params["num_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if hasattr(model, 'optimizer'):
            opt = model.optimizer
            params["optimizer"] = {"name":opt.__class__.__name__,"lr":opt.param_groups[0]['lr'] if hasattr(opt,'param_groups') else None}
        if hasattr(model, 'loss_fn'):
            params["loss_function"] = model.loss_fn.__class__.__name__ if hasattr(model.loss_fn,'__class__') else str(model.loss_fn)
    except Exception as e:
        params["extraction_error"] = str(e)
    return params


class PyTorchAdapter(MLAdapter):
    @property
    def framework(self) -> FrameworkType:
        return FrameworkType.PYTORCH

    def create_model(self, model_name: str, **kwargs) -> Any:
        # ... (your existing create_model code unchanged)
        if '.' in model_name:
            parts = model_name.split('.')
            cls   = parts[-1]
            modp  = '.'.join(parts[:-1])
            try:
                module = importlib.import_module(modp)
                model_cls = getattr(module, cls)
                return model_cls(**kwargs)
            except Exception as e:
                raise ModelCreationError(f"Could not import {model_name}: {e}", details={"model_name":model_name})
        if 'model_class' in kwargs:
            mc = kwargs.pop('model_class')
            if not inspect.isclass(mc) or not issubclass(mc, nn.Module):
                raise ModelCreationError("Invalid model_class", details={"model_class":str(mc)})
            return mc(**kwargs)
        if 'layers' in kwargs:
            layers = kwargs.pop('layers')
            if isinstance(layers, list) and all(isinstance(l, nn.Module) for l in layers):
                return nn.Sequential(*layers)
            else:
                raise ModelCreationError("layers must be list of nn.Module")
        for mp in ["torch.nn","torchvision.models"]:
            try:
                mod = importlib.import_module(mp)
                if hasattr(mod, model_name):
                    return getattr(mod, model_name)(**kwargs)
            except Exception:
                pass
        raise ModelCreationError(f"Model '{model_name}' not found")

    def get_model_info(self, model: Any) -> ModelInfo:
        # ... (your existing get_model_info)
        if not isinstance(model, nn.Module):
            raise ValueError("Not a torch.nn.Module")
        mtype = _get_torch_model_type(model)
        pname = model.__class__.__name__
        params = _extract_torch_model_params(model)
        feat = getattr(model, '_feature_names', [])
        ts   = datetime.now().isoformat()
        fitted = bool(getattr(model,'_is_fitted',False) or hasattr(model,'training_metrics'))
        mets = getattr(model,'training_metrics',{})
        return ModelInfo(
            name=pname,
            model_type=mtype,
            framework=FrameworkType.PYTORCH,
            params=params,
            description=(model.__doc__ or "").split("\n")[0],
            version=torch.__version__,
            created_at=ts,
            metrics=mets,
            feature_names=feat,
            target_name=None,
            serializable=True
        )

    def prepare_data(
        self,
        data: Union[DataFrameContainer, pd.DataFrame],
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[Any, Optional[Any]]:
        # ... (your existing prepare_data unchanged)
        df = data.data if isinstance(data,DataFrameContainer) else data
        if not isinstance(df,pd.DataFrame):
            raise ProcessingError(f"Expected DataFrame, got {type(df)}")
        if feature_columns:
            missing = [c for c in feature_columns if c not in df.columns]
            if missing:
                raise ProcessingError(f"Missing features: {missing}")
            X_df = df[feature_columns]
        else:
            X_df = df.drop(columns=[target_column]) if target_column else df
        X_np = X_df.to_numpy()
        if X_np.shape[1]==0:
            raise ProcessingError("No features in data")
        y_np = df[target_column].to_numpy() if target_column and target_column in df.columns else None
        dtype = getattr(torch, kwargs.get('tensor_dtype','float32'))
        X_t = torch.tensor(X_np, dtype=dtype)
        y_t = torch.tensor(y_np, dtype=dtype) if y_np is not None else None
        dev = torch.device(kwargs['device']) if 'device' in kwargs else None
        if dev:
            X_t = X_t.to(dev)
            if y_t is not None:
                y_t = y_t.to(dev)
        setattr(X_t,'_feature_names',list(X_df.columns))
        if kwargs.get('create_dataloader',False):
            bs = kwargs.get('batch_size',32)
            sh = kwargs.get('shuffle',True)
            ds = TensorDataset(X_t, y_t) if y_t is not None else TensorDataset(X_t)
            dl = DataLoader(ds, batch_size=bs, shuffle=sh)
            class NamedDataLoader:
                def __init__(self, dl, fnames):
                    self.dataloader = dl
                    self.feature_names = fnames
                    self._feature_names = fnames
                def __iter__(self): return iter(self.dataloader)
                def __len__(self):  return len(self.dataloader)
            return NamedDataLoader(dl, list(X_df.columns)), None
        return X_t, y_t

    def train(
        self,
        model: Any,
        X: Any,
        y: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Train a PyTorch model.
        """
        try:
            if not isinstance(model, nn.Module):
                raise ValueError("Model is not a PyTorch nn.Module")

            # Hyperparameters
            epochs           = kwargs.pop('epochs', 10)
            batch_size       = kwargs.pop('batch_size', 32)
            validation_split = kwargs.pop('validation_split', 0.2)
            verbose          = kwargs.pop('verbose', 1)
            device_name      = kwargs.pop('device', None)
            callbacks        = kwargs.pop('callbacks', [])

            # Device & move model
            device = torch.device(device_name) if device_name else (
                     torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            model = model.to(device)

            # Optimizer
            opt_arg   = kwargs.pop('optimizer', 'adam')
            opt_prm   = kwargs.pop('optimizer_params', {'lr':0.001})
            if isinstance(opt_arg,str):
                OptCls = getattr(optim, opt_arg.capitalize(), None)
                if OptCls is None:
                    raise ValueError(f"Unknown optimizer: {opt_arg}")
                optimizer = OptCls(model.parameters(), **opt_prm)
            elif callable(opt_arg):
                optimizer = opt_arg(model.parameters(), **opt_prm)
            else:
                optimizer = opt_arg
            model.optimizer = optimizer

            # Loss fn
            loss_arg = kwargs.pop('loss', None)
            if loss_arg is None:
                loss_arg = 'cross_entropy' if _get_torch_model_type(model)==ModelType.CLASSIFIER else 'mse'
            if isinstance(loss_arg,str):
                la = loss_arg.lower()
                if la=='mse': loss_fn = nn.MSELoss()
                elif la in ('cross_entropy','crossentropy'): loss_fn = nn.CrossEntropyLoss()
                elif la=='bce': loss_fn = nn.BCEWithLogitsLoss()
                elif la=='l1': loss_fn = nn.L1Loss()
                else:
                    LCls = getattr(nn,f"{la.upper()}Loss",None)
                    loss_fn = LCls() if LCls else (_ for _ in ()).throw(ValueError(f"Unknown loss: {loss_arg}"))
            elif callable(loss_arg):
                loss_fn = loss_arg
            else:
                loss_fn = loss_arg
            model.loss_fn = loss_fn

            # Prepare loaders
            if isinstance(X, DataLoader) or (hasattr(X,'dataloader') and isinstance(X.dataloader,DataLoader)):
                train_loader = X.dataloader if hasattr(X,'dataloader') else X
                val_loader   = None
            else:
                # convert inputs to tensors & split
                Xt = X if isinstance(X,torch.Tensor) else torch.tensor(X, dtype=torch.float32)
                yt = (y if isinstance(y,torch.Tensor) else torch.tensor(y,dtype=torch.float32)) if y is not None else None
                Xt, yt = Xt.to(device), (yt.to(device) if yt is not None else None)
                if validation_split>0:
                    N = len(Xt)
                    vi = int(N*validation_split)
                    idx = torch.randperm(N)
                    ti, vi = idx[vi:], idx[:vi]
                    Xtr,Xvl = Xt[ti], Xt[vi]
                    ytr,yvl = (yt[ti],yt[vi]) if yt is not None else (None,None)
                    tr_ds = TensorDataset(Xtr,ytr) if ytr is not None else TensorDataset(Xtr)
                    vl_ds = TensorDataset(Xvl,yvl) if yvl is not None else TensorDataset(Xvl)
                    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
                    val_loader   = DataLoader(vl_ds, batch_size=batch_size, shuffle=False)
                else:
                    ds = TensorDataset(Xt,yt) if yt is not None else TensorDataset(Xt)
                    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
                    val_loader   = None

            # History
            training_history = {'train_loss':[], 'val_loss':([] if val_loader else None)}
            model.training_metrics = {}

            # Training loop
            model.train()
            for epoch in range(epochs):
                ep_loss, nb = 0.0, 0
                for batch in train_loader:
                    inp, tgt = (batch if len(batch)==2 else (batch[0],None))
                    inp = inp.to(device)
                    tgt = tgt.to(device) if tgt is not None else None
                    optimizer.zero_grad()
                    out = model(inp)
                    loss = loss_fn(out,tgt) if tgt is not None else (out[1] if isinstance(out,tuple) and len(out)>1 else out)
                    loss.backward()
                    optimizer.step()
                    ep_loss += loss.item()
                    nb += 1
                avg_tr = ep_loss/max(nb,1)
                training_history['train_loss'].append(avg_tr)

                if val_loader:
                    model.eval()
                    vl_loss, vb = 0.0, 0
                    with torch.no_grad():
                        for batch in val_loader:
                            inp, tgt = (batch if len(batch)==2 else (batch[0],None))
                            inp = inp.to(device)
                            tgt = tgt.to(device) if tgt is not None else None
                            out = model(inp)
                            loss = loss_fn(out,tgt) if tgt is not None else (out[1] if isinstance(out,tuple) and len(out)>1 else out)
                            vl_loss += loss.item()
                            vb += 1
                    avg_vl = vl_loss/max(vb,1)
                    training_history['val_loss'].append(avg_vl)
                    model.train()

                # <<< Completed verbose & callbacks integration >>>
                if verbose>0:
                    if val_loader:
                        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_tr:.4f}, Val Loss: {avg_vl:.4f}")
                    else:
                        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_tr:.4f}")
                for cb in callbacks:
                    try:
                        cb(model=model, epoch=epoch+1, train_loss=avg_tr, val_loss=(avg_vl if val_loader else None))
                    except Exception as cberr:
                        print(f"Callback error at epoch {epoch+1}: {cberr}")

            model.training_metrics.update(training_history)
            model._is_fitted = True
            return model

        except Exception as e:
            if isinstance(e, TrainingError):
                raise
            raise TrainingError(f"Training failed at epoch {epoch+1}: {e}", details={"original_error": str(e)})

    def predict(
        self,
        model: Any,
        X: Any,
        **kwargs
    ) -> Any:
        """
        Make predictions with a PyTorch model.
        
        Args:
            model: The trained PyTorch model
            X: Feature data (tensor or dataset)
            **kwargs: Additional prediction parameters
            
        Returns:
            Model predictions as numpy array
            
        Raises:
            PredictionError: If prediction fails
        """
        try:
            if not isinstance(model, nn.Module):
                raise ValueError("Model is not a PyTorch nn.Module")
            
            # Check if model is fitted
            if not hasattr(model, '_is_fitted') or not model._is_fitted:
                raise PredictionError("Cannot predict with untrained model")
            
            # Get device
            device_name = kwargs.pop('device', None)
            if device_name is None and hasattr(model, 'parameters'):
                try:
                    device = next(model.parameters()).device
                except StopIteration:
                    device = torch.device('cpu')
            else:
                device = torch.device(device_name) if device_name else torch.device('cpu')
            
            # Set model to evaluation mode
            model.eval()
            
            # Check if model has custom predict method
            if hasattr(model, 'predict') and callable(model.predict):
                with torch.no_grad():
                    if isinstance(X, torch.Tensor):
                        X = X.to(device)
                        output = model.predict(X)
                    else:
                        # Handle different input types
                        output = model.predict(X)
                    return output
            
            # Handle DataLoader or similar wrapper
            if hasattr(X, 'dataloader') and isinstance(X.dataloader, DataLoader):
                all_preds = []
                with torch.no_grad():
                    for batch in X.dataloader:
                        # DataLoader may return tuples with (inputs, targets)
                        if isinstance(batch, (list, tuple)) and len(batch) > 0:
                            batch_x = batch[0]
                        else:
                            batch_x = batch
                        
                        batch_x = batch_x.to(device)
                        outputs = model(batch_x)
                        
                        # Handle different output formats
                        if isinstance(outputs, (list, tuple)):
                            outputs = outputs[0]  # Take first element for predictions
                        
                        # Get class predictions for classification models
                        if model._model_type == ModelType.CLASSIFIER or outputs.shape[-1] > 1:
                            _, preds = torch.max(outputs, 1)
                        else:
                            preds = outputs
                        
                        all_preds.append(preds.cpu())
                
                return torch.cat(all_preds)
            
            # Standard tensor input
            if isinstance(X, torch.Tensor):
                X = X.to(device)
                
                with torch.no_grad():
                    outputs = model(X)
                    
                    # Handle different output formats
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]
                    
                    # Get class predictions for classification models
                    if (_get_torch_model_type(model) == ModelType.CLASSIFIER or 
                        outputs.shape[-1] > 1):
                        _, preds = torch.max(outputs, 1)
                    else:
                        preds = outputs
                    
                    return preds
            
            # Other input types
            raise ValueError(f"Unsupported input type for prediction: {type(X)}")
            
        except Exception as e:
            if isinstance(e, PredictionError):
                raise
            raise PredictionError(f"Prediction failed: {e}", details={"original_error": str(e)})
    
    def evaluate(
        self,
        model: Any,
        X: Any,
        y: Any,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate a PyTorch model's performance.
        
        Args:
            model: The trained PyTorch model
            X: Feature data
            y: True target values
            metrics: List of metric names to compute
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of metric names to values
            
        Raises:
            EvaluationError: If evaluation fails
        """
        try:
            # Default metrics
            if metrics is None:
                model_type = _get_torch_model_type(model)
                if model_type in (ModelType.CLASSIFIER, ModelType.NEURAL_NETWORK):
                    metrics = ["accuracy", "precision", "recall", "f1"]
                else:  # Regression
                    metrics = ["mse", "rmse", "mae", "r2"]
            
            # Make predictions
            y_pred = self.predict(model, X, **kwargs)
            
            # Convert tensors to numpy arrays for metric calculation
            if isinstance(y, torch.Tensor):
                y_true = y.cpu().numpy()
            else:
                y_true = np.asarray(y)
                
            if isinstance(y_pred, torch.Tensor):
                y_pred_np = y_pred.cpu().numpy()
            else:
                y_pred_np = np.asarray(y_pred)
            
            # Initialize results dictionary
            results = {}
            
            # Process metrics - some may be custom callables
            for metric in metrics:
                try:
                    if callable(metric):
                        # Handle custom callable metrics
                        metric_name = metric.__name__
                        metric_value = metric(y_true, y_pred_np)
                        results[metric_name] = float(metric_value)
                    elif isinstance(metric, str):
                        # Handle standard metrics by name
                        if metric in ("accuracy", "balanced_accuracy"):
                            from sklearn.metrics import accuracy_score, balanced_accuracy_score
                            if metric == "accuracy":
                                results[metric] = float(accuracy_score(y_true, y_pred_np))
                            else:
                                results[metric] = float(balanced_accuracy_score(y_true, y_pred_np))
                        
                        # Classification metrics
                        elif metric in ("precision", "recall", "f1"):
                            from sklearn.metrics import precision_score, recall_score, f1_score
                            # Handle binary and multiclass
                            try:
                                if metric == "precision":
                                    results[metric] = float(precision_score(y_true, y_pred_np, average='macro'))
                                elif metric == "recall":
                                    results[metric] = float(recall_score(y_true, y_pred_np, average='macro'))
                                elif metric == "f1":
                                    results[metric] = float(f1_score(y_true, y_pred_np, average='macro'))
                            except Exception as e:
                                results[f"{metric}_error"] = str(e)
                        
                        # ROC AUC for binary classification
                        elif metric == "roc_auc":
                            try:
                                from sklearn.metrics import roc_auc_score
                                # Need to check if binary or multiclass
                                if len(np.unique(y_true)) == 2:
                                    # Binary classification
                                    results[metric] = float(roc_auc_score(y_true, y_pred_np))
                                else:
                                    results[f"{metric}_error"] = "ROC AUC not supported for multiclass without probabilities"
                            except Exception as e:
                                results[f"{metric}_error"] = str(e)
                        
                        # Confusion matrix
                        elif metric == "confusion_matrix":
                            from sklearn.metrics import confusion_matrix
                            cm = confusion_matrix(y_true, y_pred_np)
                            results[metric] = cm.tolist()
                        
                        # Regression metrics
                        elif metric in ("mse", "rmse", "mae", "r2"):
                            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                            if metric == "mse":
                                results[metric] = float(mean_squared_error(y_true, y_pred_np))
                            elif metric == "rmse":
                                results[metric] = float(np.sqrt(mean_squared_error(y_true, y_pred_np)))
                            elif metric == "mae":
                                results[metric] = float(mean_absolute_error(y_true, y_pred_np))
                            elif metric == "r2":
                                results[metric] = float(r2_score(y_true, y_pred_np))
                        
                        # Additional regression metrics
                        elif metric == "explained_variance":
                            from sklearn.metrics import explained_variance_score
                            results[metric] = float(explained_variance_score(y_true, y_pred_np))
                        
                        elif metric == "max_error":
                            from sklearn.metrics import max_error
                            results[metric] = float(max_error(y_true, y_pred_np))
                        
                        elif metric == "median_absolute_error":
                            from sklearn.metrics import median_absolute_error
                            results[metric] = float(median_absolute_error(y_true, y_pred_np))
                        
                        # Adjusted R²
                        elif metric == "adjusted_r2":
                            from sklearn.metrics import r2_score
                            n = len(y_true)
                            if isinstance(X, torch.Tensor):
                                p = X.shape[1]  # Number of features
                            else:
                                p = X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else 1
                            r2 = r2_score(y_true, y_pred_np)
                            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                            results[metric] = float(adj_r2)
                        
                        # Matthews correlation coefficient
                        elif metric == "matthews_corrcoef":
                            from sklearn.metrics import matthews_corrcoef
                            results[metric] = float(matthews_corrcoef(y_true, y_pred_np))
                        
                        # Jaccard similarity
                        elif metric == "jaccard_score":
                            from sklearn.metrics import jaccard_score
                            try:
                                results[metric] = float(jaccard_score(y_true, y_pred_np, average='macro'))
                            except Exception as e:
                                results[f"{metric}_error"] = str(e)
                        
                        # Fall back to other sklearn metrics
                        else:
                            try:
                                import sklearn.metrics as sk_metrics
                                metric_func = getattr(sk_metrics, metric, None)
                                if metric_func and callable(metric_func):
                                    # Try to call the metric function
                                    results[metric] = float(metric_func(y_true, y_pred_np))
                                else:
                                    results[f"{metric}_error"] = f"Metric {metric} not found in sklearn.metrics"
                            except Exception as e:
                                results[f"{metric}_error"] = str(e)
                
                except Exception as e:
                    results[f"{getattr(metric, '__name__', metric)}_error"] = str(e)
            
            return results
            
        except Exception as e:
            if isinstance(e, EvaluationError):
                raise
            raise EvaluationError(f"Evaluation failed: {e}", details={"original_error": str(e)})
    
    def save_model(self, model: Any, path: str) -> None:
        """
        Save a PyTorch model to disk.
        
        Args:
            model: The PyTorch model to save
            path: Path where the model should be saved
            
        Raises:
            SerializationError: If saving fails
        """
        try:
            if not isinstance(model, nn.Module):
                raise ValueError("Model is not a PyTorch nn.Module")
            
            # Prepare model directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # Create a dictionary with model state and metadata
            state_dict = {
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'model_module': model.__class__.__module__,
                'framework_version': torch.__version__,
                'save_time': datetime.now().isoformat()
            }
            
            # Save optimizer state if available
            if hasattr(model, 'optimizer') and model.optimizer is not None:
                state_dict['optimizer_state_dict'] = model.optimizer.state_dict()
                state_dict['optimizer_class'] = model.optimizer.__class__.__name__
            
            # Save loss function if available
            if hasattr(model, 'loss_fn') and model.loss_fn is not None:
                if isinstance(model.loss_fn, nn.Module):
                    state_dict['loss_fn_state_dict'] = model.loss_fn.state_dict()
                state_dict['loss_fn_class'] = (model.loss_fn.__class__.__name__ 
                                              if hasattr(model.loss_fn, '__class__') 
                                              else str(model.loss_fn))
            
            # Save training metrics if available
            if hasattr(model, 'training_metrics') and model.training_metrics:
                state_dict['training_metrics'] = model.training_metrics
            
            # Save feature names if available
            if hasattr(model, '_feature_names'):
                state_dict['feature_names'] = model._feature_names
            
            # Save fitted flag
            if hasattr(model, '_is_fitted'):
                state_dict['is_fitted'] = model._is_fitted
            
            # Save model type if available
            model_type = _get_torch_model_type(model)
            state_dict['model_type'] = model_type.value
            
            # Save extra custom attributes
            extra_attrs = {}
            for attr_name in dir(model):
                if attr_name.startswith('_') and attr_name not in ('_feature_names', '_is_fitted'):
                    continue
                if attr_name in ('optimizer', 'loss_fn', 'training_metrics'):
                    continue
                attr = getattr(model, attr_name)
                # Only save simple types that can be easily serialized
                if isinstance(attr, (str, int, float, bool, list, dict, tuple)) and not callable(attr):
                    extra_attrs[attr_name] = attr
            
            state_dict['extra_attributes'] = extra_attrs
            
            # Save the model using torch.save
            torch.save(state_dict, path)
            
        except Exception as e:
            if isinstance(e, SerializationError):
                raise
            raise SerializationError(f"Failed to save model: {e}", details={"original_error": str(e)})
    
    def load_model(self, path: str) -> Any:
        """
        Load a PyTorch model from disk.
        
        Args:
            path: Path from which to load the model
            
        Returns:
            The loaded PyTorch model
            
        Raises:
            SerializationError: If loading fails
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            
            # Load the state dictionary
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            
            # Extract model class and module information
            model_class_name = state_dict.get('model_class')
            model_module_name = state_dict.get('model_module')
            
            # Try to recreate the model class
            model_cls = None
            if model_module_name and model_class_name:
                try:
                    # Try to import the module and get the class
                    module = importlib.import_module(model_module_name)
                    model_cls = getattr(module, model_class_name, None)
                except ImportError:
                    # If module import fails, attempt to find the class in standard modules
                    for module_path in ["torch.nn", "torchvision.models"]:
                        try:
                            module = importlib.import_module(module_path)
                            if hasattr(module, model_class_name):
                                model_cls = getattr(module, model_class_name)
                                break
                        except ImportError:
                            pass
            
            # If we couldn't find the model class, raise an error
            if model_cls is None:
                raise SerializationError(
                    f"Could not locate model class '{model_class_name}' in module '{model_module_name}'",
                    details={
                        "model_class": model_class_name,
                        "model_module": model_module_name
                    }
                )
            
            # Create an instance of the model
            model = model_cls()
            
            # Load the model state
            model.load_state_dict(state_dict['model_state_dict'])
            
            # Set the model to evaluation mode
            model.eval()
            
            # Restore optimizer if available
            if 'optimizer_state_dict' in state_dict and 'optimizer_class' in state_dict:
                try:
                    optimizer_class_name = state_dict['optimizer_class']
                    optimizer_cls = getattr(optim, optimizer_class_name)
                    model.optimizer = optimizer_cls(model.parameters())
                    model.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
                except Exception as opt_error:
                    # Log the error but continue
                    print(f"Warning: Could not restore optimizer: {opt_error}")
            
            # Restore loss function if available
            if 'loss_fn_class' in state_dict:
                try:
                    loss_fn_class_name = state_dict['loss_fn_class']
                    
                    # Try to recreate the loss function
                    if loss_fn_class_name in dir(nn):
                        loss_cls = getattr(nn, loss_fn_class_name)
                        model.loss_fn = loss_cls()
                        
                        # If loss function has state, restore it
                        if 'loss_fn_state_dict' in state_dict and isinstance(model.loss_fn, nn.Module):
                            model.loss_fn.load_state_dict(state_dict['loss_fn_state_dict'])
                    else:
                        # For built-in or custom loss functions, store the name
                        model.loss_fn = loss_fn_class_name
                except Exception as loss_error:
                    # Log the error but continue
                    print(f"Warning: Could not restore loss function: {loss_error}")
            
            # Restore training metrics
            if 'training_metrics' in state_dict:
                model.training_metrics = state_dict['training_metrics']
            
            # Restore feature names
            if 'feature_names' in state_dict:
                model._feature_names = state_dict['feature_names']
            
            # Restore fitted flag
            if 'is_fitted' in state_dict:
                model._is_fitted = state_dict['is_fitted']
            else:
                # Default to True since we're loading a saved model
                model._is_fitted = True
            
            # Restore model type
            if 'model_type' in state_dict:
                try:
                    model._model_type = ModelType(state_dict['model_type'])
                except (ValueError, KeyError):
                    # If model_type can't be converted, infer it
                    model._model_type = _get_torch_model_type(model)
            
            # Restore extra attributes
            if 'extra_attributes' in state_dict and isinstance(state_dict['extra_attributes'], dict):
                for key, value in state_dict['extra_attributes'].items():
                    if not key.startswith('__'):  # Skip any dunder attributes
                        setattr(model, key, value)
            
            return model
            
        except Exception as e:
            if isinstance(e, SerializationError):
                raise
            raise SerializationError(f"Failed to load model: {e}", details={"original_error": str(e)})
    
    def create_pipeline_transformer(self, model: Any, **kwargs) -> Any:
        """
        Create a Prometheum transformer from this model.
        
        Args:
            model: The trained PyTorch model
            **kwargs: Additional parameters for the transformer
            
        Returns:
            A transformer that can be used in a Prometheum pipeline
            
        Raises:
            ProcessingError: If transformer creation fails
        """
        try:
            if not isinstance(model, nn.Module):
                raise ValueError("Model is not a PyTorch nn.Module")
                
            if not hasattr(model, '_is_fitted') or not model._is_fitted:
                raise ValueError("Cannot create transformer from untrained model")
            
            # Create a transformer that wraps the PyTorch model
            class PyTorchTransformer(DataTransformer):
                def __init__(self, pytorch_model, adapter, **config):
                    super().__init__(**config)
                    self.model = pytorch_model
                    self.adapter = adapter
                    
                def transform(self, data):
                    # Prepare the data
                    X, _ = self.adapter.prepare_data(data)
                    
                    # Make predictions
                    predictions = self.adapter.predict(self.model, X)
                    
                    # Convert predictions to DataFrame
                    if isinstance(predictions, torch.Tensor):
                        predictions = predictions.cpu().numpy()
                    
                    # Create prediction column name
                    pred_col = kwargs.get('prediction_column', 'prediction')
                    
                    # Add predictions to the data
                    result_df = data.copy()
                    result_df[pred_col] = predictions
                    
                    return result_df
                
                def get_info(self):
                    return {
                        "type": "PyTorchTransformer",
                        "model_info": self.adapter.get_model_info(self.model).to_dict()
                    }
            
            return PyTorchTransformer(model, self, **kwargs)
            
        except Exception as e:
            raise ProcessingError(f"Failed to create transformer: {e}", details={"original_error": str(e)})
