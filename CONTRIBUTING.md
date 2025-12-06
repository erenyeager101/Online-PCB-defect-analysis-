# Contributing to PCB Defect Detection

First off, thank you for considering contributing to PCB Defect Detection! It's people like you that make this project better for everyone.

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## 🤔 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

**Bug Report Template:**
- **Title**: Clear and descriptive title
- **Description**: Detailed description of the issue
- **Steps to Reproduce**: 
  1. Step 1
  2. Step 2
  3. ...
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Screenshots**: If applicable
- **Environment**:
  - OS: [e.g., Windows 10, Ubuntu 20.04]
  - Python Version: [e.g., 3.8.5]
  - TensorFlow Version: [e.g., 2.8.0]
  - Browser (if web-related): [e.g., Chrome 98]

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description**
- **Use case**: Why is this enhancement useful?
- **Current behavior vs. proposed behavior**
- **Possible implementation**: If you have ideas
- **Mockups/Examples**: If applicable

### Pull Requests

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Defect-Detection-of-PCB.git
   cd Defect-Detection-of-PCB
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/YourFeatureName
   # or
   git checkout -b fix/YourBugFix
   ```

3. **Set Up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8  # Development dependencies
   ```

4. **Make Your Changes**
   - Write clear, commented code
   - Follow the coding standards (see below)
   - Add tests if applicable
   - Update documentation as needed

5. **Test Your Changes**
   ```bash
   # Run tests
   pytest tests/
   
   # Check code style
   black --check .
   flake8 .
   
   # Run the smoke test
   python smoke_test.py
   ```

6. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: Add amazing feature"
   ```
   
   **Commit Message Format:**
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code style changes (formatting, etc.)
   - `refactor:` Code refactoring
   - `test:` Adding or updating tests
   - `chore:` Maintenance tasks

7. **Push to Your Fork**
   ```bash
   git push origin feature/YourFeatureName
   ```

8. **Create Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill in the PR template with details

## 💻 Development Guidelines

### Code Style

- **Python**: Follow PEP 8 guidelines
- **Line Length**: Maximum 100 characters
- **Formatting**: Use `black` for automatic formatting
- **Imports**: Organize imports (standard library, third-party, local)
- **Docstrings**: Use Google-style docstrings

**Example:**
```python
def predict_defect(image_path: str) -> dict:
    """
    Predict whether a PCB image contains defects.
    
    Args:
        image_path: Path to the PCB image file.
        
    Returns:
        Dictionary containing prediction results with keys:
            - 'defective': bool
            - 'confidence': float
            - 'processing_time': float
            
    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If image format is not supported.
    """
    pass
```

### Testing

- Write unit tests for new features
- Maintain or improve code coverage
- Test edge cases and error handling
- Use meaningful test names

**Example:**
```python
def test_predict_defect_with_valid_image():
    """Test prediction with a valid PCB image."""
    result = predict_defect('tests/data/good_pcb.jpg')
    assert 'defective' in result
    assert isinstance(result['confidence'], float)
```

### Documentation

- Update README.md if adding features
- Add docstrings to all functions/classes
- Include inline comments for complex logic
- Update API documentation if applicable

### File Organization

```
New Feature Example:
├── app.py                 # Update if adding routes
├── notebooks/
│   └── feature_demo.ipynb # Demo notebook for new feature
├── tests/
│   └── test_feature.py   # Unit tests
└── README.md             # Update documentation
```

## 🎯 Priority Areas

We especially welcome contributions in these areas:

1. **Model Improvements**
   - Multi-class defect classification
   - Model optimization and quantization
   - Explainable AI (Grad-CAM, LIME)

2. **Web Application**
   - UI/UX enhancements
   - Real-time processing
   - RESTful API development
   - Mobile responsiveness

3. **Testing**
   - Increase test coverage
   - Integration tests
   - Performance benchmarks

4. **Documentation**
   - Tutorial notebooks
   - API documentation
   - Deployment guides

5. **DevOps**
   - Docker containerization
   - CI/CD pipeline
   - Cloud deployment guides

## 🔍 Review Process

1. **Automated Checks**: CI/CD runs tests and linting
2. **Code Review**: Maintainers review code quality and design
3. **Testing**: Verify functionality works as expected
4. **Documentation**: Ensure docs are updated
5. **Approval**: At least one maintainer approval required
6. **Merge**: Squash and merge into main branch

## 📋 Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts with main branch
- [ ] Screenshots included (if UI changes)
- [ ] Performance impact considered

## 🐛 Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information requested
- `wontfix`: This will not be worked on
- `duplicate`: This issue already exists

## 💡 Tips for Success

- **Start Small**: Begin with small changes to understand the codebase
- **Ask Questions**: Don't hesitate to ask in issues or discussions
- **Be Patient**: Reviews may take time
- **Be Respectful**: We're all learning together
- **Have Fun**: Enjoy the process!

## 🔗 Useful Resources

- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [Python PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [TensorFlow Best Practices](https://www.tensorflow.org/guide/effective_tf2)
- [Flask Documentation](https://flask.palletsprojects.com/)

## 📞 Getting Help

- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: For bugs and feature requests
- **Documentation**: Check README and docs folder

## 🙏 Recognition

Contributors will be:
- Listed in the Contributors section
- Mentioned in release notes
- Given credit in commits

Thank you for contributing! 🎉
