

# 🤝 Contributing to AnomaVision

First off, thanks for taking the time to contribute! 🎉
We welcome all kinds of contributions — whether it’s bug reports, feature requests, documentation improvements, or new code.

---

## 📌 How to Contribute

1. **Fork the repo** and create your branch from `main`:

   ```bash
   git checkout -b feature/your-feature
   ```
2. **Install in development mode**:

   ```bash
   poetry install --extras "full"
   poetry shell
   pip install -e .[dev]
   ```
3. **Write tests** for new functionality.
4. **Run tests & linters** before pushing:

   ```bash
   pytest tests/
   black .
   isort .
   flake8 .
   ```
5. **Commit with clear messages** (see below).
6. **Open a Pull Request (PR)** and describe your changes.

---

## 🧪 Code Style & Quality

We follow these standards:

* **Python ≥ 3.9**
* **Black** for formatting
* **isort** for import ordering
* **flake8** for linting
* **pytest** for testing

Run all checks with:

```bash
make lint test
```

---

## 📝 Commit Messages

Use clear and descriptive commit messages.
Format:

```
<type>(<scope>): <message>
```

Examples:

* `fix(detect): handle empty directory gracefully`
* `feat(export): add OpenVINO INT8 support`
* `docs(api): improve Padim usage examples`

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

---

## 🔀 Pull Request Guidelines

* Keep PRs focused and small.
* Update docs if behavior changes.
* Ensure all tests pass (`pytest`).
* Add benchmarks if performance-critical.

---

## 🐛 Reporting Bugs

* Use the [GitHub Issues](https://github.com/DeepKnowledge1/AnomaVision/issues).
* Include:

  * OS & Python version
  * Installation method (pip/poetry)
  * Steps to reproduce
  * Expected vs actual behavior

---

## 💡 Suggesting Features

* Open a [Discussion](https://github.com/DeepKnowledge1/AnomaVision/discussions).
* Clearly describe **why** the feature is useful and how it fits into the project.

---

## 🧑‍💻 Development Workflow

Typical workflow for contributors:

```bash
# Clone your fork
git clone https://github.com/<your-username>/AnomaVision.git
cd AnomaVision

# Create a branch
git checkout -b fix/bug-name

# Make changes and commit
git commit -m "fix(detect): corrected threshold application"

# Push and open PR
git push origin fix/bug-name
```

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

Big thanks to all contributors who help make AnomaVision better for the community! 🚀

