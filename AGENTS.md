## 📌 Role: Enhanced Code Quality for `pyear`

This document guides AI agents (e.g., OpenAI Codex, ChatGPT) in generating consistently high-quality, production-ready code aligned precisely with the standards of the `pyear` project, which specializes in fatigue detection and analysis of blink-related ocular signals.

---

## 🎯 Goals

* ✅ Accelerate AI onboarding into the `pyear` codebase
* ✅ Ensure consistency and adherence to project standards
* ✅ Minimize the necessity of extensive manual review
* ✅ Improve overall code reliability, maintainability, and testability

---


## 📌 Key Coding Practices

### 📄 Input Handling and Testing


* **Testing Inputs**:

  * Use provided mock EAR signals (`unittest.fixtures.mock_ear_generation`).
  * Ensure tests comprehensively cover both continuous and epoch scenarios.

### 📂 Feature Implementation Guidelines

* **Separate Python files** for each distinct feature.

  * For example, individual files for `blink_count`, `blink_rate`, and `inter_blink_interval`.
* **Group related features within dedicated directories**.

  * Example: All blink event metrics under `blink_events`.
* **Dedicated unit tests for each feature file**.

  * Facilitates easy debugging, clarity, and maintainability.

**Feature selection flexibility**:

  -   Allow users to select one or multiple specific features to calculate.

  -   By default, assume the user intends to extract all available features.
### 🪵 Logging Standards

* Define root logger configuration in `pipeline.py`
* Each Python module initializes: `logger = logging.getLogger(__name__)`
* Include per-function entry and exit logs (`INFO` level) and critical internal states (`DEBUG` level).

### ⏳ Progress Feedback

* Consistently employ progress bars (`from tqdm import tqdm`) for substantial iteration processes (epochs, signals).

### 📄 Docstrings and Type Hints

* Use clear, structured docstrings (Google or reStructuredText format).
* Explicitly document parameters, returns, and exceptions.
* Type annotations (`typing`) are mandatory for clarity and maintainability.
* Detail docstrings for all public functions and uni test, including:

  * **Parameters**: Types and descriptions.
  * **Returns**: Expected types and structure.
  * **Raises**: Document any exceptions that may be raised.
### 🧪 Comprehensive Unit Testing

* Thorough testing using `pytest` or Python’s `unittest`.
* Separate, descriptive test files within the `unittest/` directory.
* Employ synthetic EAR signals provided by mocks to simulate realistic data scenarios.

---

## ✅ Conventions for Consistency

| Aspect             | Standard                                          |
| ------------------ | ------------------------------------------------- |
| Function Naming    | `snake_case`                                      |
| Class Naming       | `PascalCase`                                      |
| Variable Naming    | `snake_case`                                      |
| Imports            | Grouped logically: standard → third-party → local |
| Sampling Frequency | Standard at 30 Hz unless specified otherwise      |
| DataFrame Outputs  | Clearly named and structured columns              |

---

## 🧩 Modularization Guidance

* Modules must be self-contained, individually testable, and clearly defined.
* Minimize cross-module dependencies and avoid side-effects.
* Accept inputs as raw arrays or MNE objects; consistently return structured DataFrames or clear intermediate outputs.
## Rename variables to PIP conventions

For unit testing and remaping, import the following:

```python
from .utils.update_pkl_variables import RENAME_MAP
```
at the end, run
ruff check .