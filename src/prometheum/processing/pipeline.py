"""
Pipeline orchestration system for the Prometheum framework.

This module provides classes for defining, building, and executing data processing 
pipelines that chain together multiple transformation steps.
"""

import copy
import datetime
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import pandas as pd

from prometheum.core.base import DataContainer, DataFrameContainer, DataProcessor, DataTransformer, Pipeline as CorePipeline
from prometheum.core.exceptions import PipelineError, ProcessingError


class StepStatus(Enum):
    """Enumeration of possible step execution statuses."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class ExecutionContext:
    """Context for tracking pipeline execution state and metadata."""
    
    def __init__(self) -> None:
        """Initialize the execution context."""
        self.start_time = None
        self.end_time = None
        self.steps_executed = 0
        self.steps_skipped = 0
        self.steps_failed = 0
        self.step_history = []
        self.current_step = None
        self.execution_metadata = {}
    
    def start_execution(self) -> None:
        """Mark the start of pipeline execution."""
        self.start_time = datetime.datetime.now()
        self.execution_metadata["started_at"] = self.start_time.isoformat()
    
    def end_execution(self) -> None:
        """Mark the end of pipeline execution."""
        self.end_time = datetime.datetime.now()
        self.execution_metadata["ended_at"] = self.end_time.isoformat()
        
        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
            self.execution_metadata["duration_seconds"] = duration
    
    def start_step(self, step_name: str, step_index: int) -> None:
        """Mark the start of a pipeline step.
        
        Args:
            step_name: Name of the step
            step_index: Index of the step in the pipeline
        """
        self.current_step = {
            "name": step_name,
            "index": step_index,
            "status": StepStatus.RUNNING.value,
            "started_at": datetime.datetime.now().isoformat()
        }
    
    def end_step(self, status: StepStatus, error: Optional[Exception] = None) -> None:
        """Mark the end of a pipeline step.
        
        Args:
            status: Final status of the step
            error: Optional error if the step failed
        """
        if self.current_step:
            self.current_step["status"] = status.value
            self.current_step["ended_at"] = datetime.datetime.now().isoformat()
            
            if error:
                self.current_step["error"] = str(error)
                self.current_step["error_type"] = error.__class__.__name__
            
            self.step_history.append(self.current_step)
            
            # Update counters
            if status == StepStatus.COMPLETED:
                self.steps_executed += 1
            elif status == StepStatus.SKIPPED:
                self.steps_skipped += 1
            elif status == StepStatus.FAILED:
                self.steps_failed += 1
            
            self.current_step = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution context to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the execution context
        """
        result = {
            "steps_executed": self.steps_executed,
            "steps_skipped": self.steps_skipped,
            "steps_failed": self.steps_failed,
            "step_history": self.step_history,
            **self.execution_metadata
        }
        
        return result


class PipelineStep:
    """A named step in a data processing pipeline."""
    
    def __init__(
        self,
        processor: DataProcessor,
        name: Optional[str] = None,
        condition: Optional[Callable[[DataContainer], bool]] = None
    ) -> None:
        """Initialize a pipeline step.
        
        Args:
            processor: The data processor to execute
            name: Optional name for the step (defaults to processor class name)
            condition: Optional condition function that determines if the step should execute
        """
        self.processor = processor
        self.name = name or processor.__class__.__name__
        self.condition = condition
    
    def should_execute(self, data: DataContainer) -> bool:
        """Check if the step should execute based on the condition.
        
        Args:
            data: The data to check against the condition
            
        Returns:
            bool: True if the step should execute, False otherwise
        """
        if self.condition is None:
            return True
        
        try:
            return self.condition(data)
        except Exception:
            # If condition evaluation fails, default to executing the step
            return True


class Pipeline(CorePipeline):
    """Advanced pipeline for chaining data processors with additional features."""
    
    def __init__(
        self,
        steps: Optional[List[Union[DataProcessor, Tuple[str, DataProcessor], PipelineStep]]] = None,
        name: str = "Pipeline",
        raise_on_error: bool = True,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the pipeline.
        
        Args:
            steps: Optional list of steps to add to the pipeline
            name: Name of the pipeline
            raise_on_error: Whether to raise an exception on step failure
            config: Optional configuration parameters
        """
        super().__init__([], config)
        self.name = name
        self.raise_on_error = raise_on_error
        self._steps = []  # List of PipelineStep objects
        
        # Add steps if provided
        if steps:
            for step in steps:
                if isinstance(step, DataProcessor):
                    self.add_step(step)
                elif isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], str) and isinstance(step[1], DataProcessor):
                    self.add_step(step[1], step[0])
                elif isinstance(step, PipelineStep):
                    self._steps.append(step)
                else:
                    raise ValueError(f"Invalid step: {step}. Must be DataProcessor, (name, DataProcessor) tuple, or PipelineStep.")
    
    @property
    def steps(self) -> List[PipelineStep]:
        """Get the pipeline steps.
        
        Returns:
            List[PipelineStep]: List of pipeline steps
        """
        return self._steps
    
    def add_step(
        self,
        processor: DataProcessor,
        name: Optional[str] = None,
        condition: Optional[Callable[[DataContainer], bool]] = None
    ) -> "Pipeline":
        """Add a step to the pipeline.
        
        Args:
            processor: The data processor to add
            name: Optional name for the step
            condition: Optional condition function
            
        Returns:
            Pipeline: Self for method chaining
        """
        self._steps.append(PipelineStep(processor, name, condition))
        return self
    
    def process(self, data: DataContainer) -> DataContainer:
        """Process data through the pipeline.
        
        Args:
            data: Input data to process
            
        Returns:
            DataContainer: The processed data
            
        Raises:
            PipelineError: If a step fails and raise_on_error is True
        """
        # Create execution context
        context = ExecutionContext()
        context.start_execution()
        
        # Initialize metadata
        pipeline_metadata = {
            "pipeline": {
                "name": self.name,
                "id": self.id,
                "step_count": len(self._steps)
            }
        }
        
        current_data = data
        
        # Add a way to share state between steps if needed
        shared_state = {}
        
        try:
            # Process each step
            for i, step in enumerate(self._steps):
                context.start_step(step.name, i)
                
                try:
                    # Check if step should execute
                    if step.should_execute(current_data):
                        # Execute the processor
                        start_time = time.time()
                        current_data = step.processor.process(current_data)
                        end_time = time.time()
                        
                        # Record execution time
                        step_time = end_time - start_time
                        context.current_step["execution_time"] = step_time
                        
                        context.end_step(StepStatus.COMPLETED)
                    else:
                        # Skip step
                        context.end_step(StepStatus.SKIPPED)
                
                except Exception as e:
                    # Record failure
                    context.end_step(StepStatus.FAILED, e)
                    
                    if self.raise_on_error:
                        # Prepare error message with step context
                        step_idx = i
                        step_name = step.name
                        step_id = getattr(step.processor, 'id', None)
                        
                        if isinstance(e, ProcessingError):
                            # Re-wrap as PipelineError
                            raise PipelineError(
                                f"Pipeline step '{step_name}' failed: {str(e)}",
                                pipeline_id=self.id,
                                step_idx=step_idx,
                                step_id=step_id,
                                details={
                                    "step_name": step_name,
                                    "original_error": str(e),
                                    "execution_context": context.to_dict()
                                }
                            ) from e
                        else:
                            # Wrap other exceptions as PipelineError
                            raise PipelineError(
                                f"Pipeline step '{step_name}' failed with {e.__class__.__name__}: {str(e)}",
                                pipeline_id=self.id,
                                step_idx=step_idx,
                                step_id=step_id
                            ) from e
        
        finally:
            # Finish execution context
            context.end_execution()
            
            # Add execution context to metadata
            pipeline_metadata["execution"] = context.to_dict()
            
            # Merge metadata into current data
            if hasattr(current_data, 'metadata'):
                current_data.metadata.update(pipeline_metadata)
        
        return current_data
    
    def fit_transform(self, data: DataContainer) -> DataContainer:
        """Fit all transformer steps and transform the data.
        
        This method fits all transformer steps in the pipeline and then transforms
        the data using the fitted transformers.
        
        Args:
            data: Input data to fit and transform
            
        Returns:
            DataContainer: The transformed data
            
        Raises:
            PipelineError: If fitting or transformation fails
        """
        # First fit all transformer steps
        current_data = copy.deepcopy(data)  # Copy to avoid modifying input data during fitting
        
        for i, step in enumerate(self._steps):
            processor = step.processor
            
            # If the processor is a transformer, fit it
            if isinstance(processor, DataTransformer):
                try:
                    processor.fit(current_data)
                except Exception as e:
                    # Wrap exception as PipelineError
                    raise PipelineError(
                        f"Failed to fit transformer '{step.name}': {str(e)}",
                        pipeline_id=self.id,
                        step_idx=i,
                        step_id=getattr(processor, 'id', None)
                    ) from e
            
            # Apply the processor to update current_data for the next fit
            if step.should_execute(current_data):
                try:
                    current_data = processor.process(current_data)
                except Exception as e:
                    # Just log the error during fitting - we'll catch it during the real process
                    pass
        
        # Now process the original data through the pipeline
        return self.process(data)
    
    def clone(self) -> "Pipeline":
        """Create a deep copy of the pipeline.
        
        Returns:
            Pipeline: A deep copy of this pipeline
        """
        return copy.deepcopy(self)


class PipelineBuilder:
    """Builder for fluent pipeline construction."""
    
    def __init__(self, name: str = "Pipeline", raise_on_error: bool = True) -> None:
        """Initialize the pipeline builder.
        
        Args:
            name: Name of the pipeline
            raise_on_error: Whether to raise an exception on step failure
        """
        self.pipeline = Pipeline(name=name, raise_on_error=raise_on_error)
    
    def add(
        self,
        processor: DataProcessor,
        name: Optional[str] = None,
        condition: Optional[Callable[[DataContainer], bool]] = None
    ) -> "PipelineBuilder":
        """Add a step to the pipeline.
        
        Args:
            processor: The data processor to add
            name: Optional name for the step
            condition: Optional condition function
            
        Returns:
            PipelineBuilder: Self for method chaining
        """
        self.pipeline.add_step(processor, name, condition)
        return self
    
    def with_config(self, config: Dict[str, Any]) -> "PipelineBuilder":
        """Set configuration for the pipeline.
        
        Args:
            config: Configuration parameters
            
        Returns:
            PipelineBuilder: Self for method chaining
        """
        self.pipeline._config.update(config)
        return self
    
    def if_true(
        self,
        condition: Callable[[DataContainer], bool],
        processors: List[DataProcessor]
    ) -> "PipelineBuilder":
        """Add conditional steps that execute only if condition is True.
        
        Args:
            condition: Condition function
            processors: List of processors to add conditionally
            
        Returns:
            PipelineBuilder: Self for method chaining
        """
        for processor in processors:
            self.add(processor, condition=condition)
        return self
    
    def branch(
        self,
        condition: Callable[[DataContainer], bool],
        if_true: List[DataProcessor],
        if_false: List[DataProcessor],
        branch_name: Optional[str] = None
    ) -> "PipelineBuilder":
        """Add a conditional branch to the pipeline.
        
        Args:
            condition: Condition function
            if_true: Processors to execute if condition is True
            if_false: Processors to execute if condition is False
            branch_name: Optional name for the branch
            
        Returns:
            PipelineBuilder: Self for method chaining
        """
        # Add true branch
        true_branch_name = f"{branch_name}_true" if branch_name else "true_branch"
        for i, processor in enumerate(if_true):
            name = f"{true_branch_name}_{i}"
            self.add(processor, name=name, condition=condition)
        
        # Add false branch
        false_branch_name = f"{branch_name}_false" if branch_name else "false_branch"
        for i, processor in enumerate(if_false):
            name = f"{false_branch_name}_{i}"
            # Use a closure to capture the current condition value
            # This prevents late binding issues with lambda
            def make_inverse_condition(cond):
                return lambda data: not cond(data)
            inverse_condition = make_inverse_condition(condition)
            self.add(processor, name=name, condition=inverse_condition)
        
        return self

    def build(self) -> Pipeline:
        """Build and return the pipeline.
        
        Returns:
            Pipeline: The constructed pipeline
        """
        return self.pipeline

