# Contributing to Lyse.ai

Thank you for your interest in contributing to Lyse.ai.

Lyse.ai is an open-source project, and contributions are welcome. You can contribute to the project through code, model training, datasets, documentation, testing, or bug reports.

## Ways to contribute

You can contribute in several areas:

* Code
* Model training
* Datasets
* Documentation
* Testing
* Bug reports
* Feature proposals
* Performance improvements

---

## Code contributions

The recommended workflow is:

```text
Fork the repository
        ↓
Create a branch
        ↓
Make your changes
        ↓
Test your changes
        ↓
Push your branch
        ↓
Open a Pull Request
        ↓
Code review
        ↓
Merge
```

Create a branch for your contribution:

```bash
git checkout -b feature/my-feature
```

After making your changes:

```bash
git add .
git commit -m "Add my feature"
git push origin feature/my-feature
```

Then open a Pull Request targeting the `main` branch.

### Pull Request guidelines

Please keep Pull Requests focused on a specific change.

A good Pull Request:

```text
Add checkpoint validation
```

A less ideal Pull Request:

```text
Add checkpoint validation + rewrite tokenizer + change model architecture + update README
```

When opening a Pull Request, explain:

* What was changed?
* Why was it changed?
* How was it tested?
* Does it affect training?
* Does it affect existing checkpoints?

Maintainers may request changes before merging.

---

## Model training contributions

If you have access to GPU resources, you can contribute compute by training Lyse.

Before starting a training run, read [`TRAINING.md`](TRAINING.md).

When submitting a training contribution, provide the following information:

```text
GPU:
GPU VRAM:
Training stage:
Configuration:
Dataset:
Number of steps:
Training loss:
Validation loss:
Checkpoint:
```

Training contributions should contain enough information for the maintainers to understand and reproduce the experiment.

Do not upload large model checkpoints directly to GitHub.

Use the project's designated model hosting platform for large checkpoint files.

---

## Dataset contributions

Dataset improvements are welcome.

Before contributing a dataset, provide:

* Dataset source
* Dataset format
* Approximate size
* License
* Preprocessing performed
* Reason why the dataset is useful for Lyse

Only contribute datasets that can legally be redistributed and used by the project.

---

## Bug reports

If you find a bug, open a GitHub Issue.

Include as much relevant information as possible:

* Description of the problem
* Expected behavior
* Actual behavior
* Steps to reproduce
* Python version
* PyTorch version
* GPU
* Relevant logs or error messages

---

## Feature requests

Feature requests are welcome.

For large changes, it is recommended to open an Issue before implementing the feature. This allows the maintainers and contributors to discuss the proposal before development begins.

---

## Testing

Always test your changes before submitting a Pull Request.

For training-related changes, perform a small test run before starting a large training job whenever possible.

---

## Experimental changes

Experiments are welcome.

If your contribution changes the training configuration, model architecture, dataset mixture, optimizer, learning rate, sequence length, or another important training parameter, document the change clearly.

This makes experiments easier to compare and reproduce.

---

## Main branch

The `main` branch is protected.

Contributors should work on their own branches and submit Pull Requests.

All Pull Requests are reviewed before being merged.

Opening a Pull Request does not guarantee that the contribution will be accepted.

---

## Code quality

Please try to:

* Keep the existing project structure
* Avoid unnecessary dependencies
* Write clear and maintainable code
* Document important changes
* Avoid unrelated modifications
* Test changes before submitting them

---

## Thank you

Thank you for helping improve Lyse.ai.

Every contribution, whether it is code, training compute, datasets, documentation, testing, or ideas, helps the project move forward.
