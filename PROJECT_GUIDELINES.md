# Prometheum Code Guidelines

This document outlines principles for maintaining clean, concise code throughout the entire Prometheum NAS Router OS project.

## Core Principles

1. **Minimize Boilerplate**: Use dependency injection and composition over inheritance to reduce repetitive code.

2. **Single Responsibility**: Each module, class, and function should have one clear purpose.

3. **Strong Typing**: Use type annotations consistently for better readability and fewer runtime errors.

4. **Error Handling**: Use centralized error handling patterns to avoid repetitive try/except blocks.

5. **Configuration Management**: Prefer environment variables and config files over hardcoded values.

## Implementation Guidelines

### Module Structure

```
prometheum/
├── src/                    # Core source code
│   ├── storage/            # Storage management
│   ├── api/                # REST API
│   ├── backup/             # Backup system
│   ├── ai/                 # AI integration
│   └── utils/              # Shared utilities
├── scripts/                # Management scripts
├── tests/                  # Test suite
└── docs/                   # Documentation
```

### Code Standards

- Use dataclasses or Pydantic models for data structures
- Prefer composition over inheritance
- Avoid deeply nested code (max 3 levels)
- Keep functions under 50 lines
- Keep files under a few hundred lines
- Use dependency injection for integration points

### API Design

- Normalize to RESTful patterns
- Use consistent data formats (JSON)
- Implement proper status codes
- Keep authentication simple but secure

### Shared Utilities

Create utilities for common operations:
- Command execution
- Filesystem operations
- Configuration management
- Error handling

### Example Pattern

```python
# Instead of:
try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.error(f"Error loading file {file_path}: {e}")
    return None

# Use a utility function:
data = load_json_file(file_path)  # Handles errors internally
```

## Implementation Notes

1. **Storage System**: Keep base abstractions minimal with filesystem-specific implementations.

2. **API Layer**: Use FastAPI with minimal middleware and dependencies.

3. **Backup System**: Focus on protocol implementations over complex scheduling.

4. **AI Integration**: Use clean adapter pattern to allow for different AI backends.

## Refactoring Priorities

1. Extract common patterns into utilities
2. Standardize error handling
3. Normalize configuration management
4. Implement consistent logging
5. Reduce duplication through shared abstractions

