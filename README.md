<<<<<<< Updated upstream
=======
[![Go Reference](https://pkg.go.dev/badge/github.com/tbiehn/gptparallel.svg)](https://pkg.go.dev/github.com/tbiehn/gptparallel)

>>>>>>> Stashed changes
# GPT Parallel

GPT Parallel is an early beta library for the Go programming language that helps you manage multiple concurrent requests to the OpenAI API and OpenAI Azure. It's built on top of the go-openai package and is designed to make it easier to handle multiple requests, track progress, and apply retries with a customizable exponential backoff strategy.

**Disclaimer:** This library is in early beta and error handling has not been thoroughly tested. Use it at your own risk.

## Features

- Supports concurrent requests to GPT models
- Progress tracking using mpb (multi progress bars)
- Customizable exponential backoff strategy for retries
- Custom logger support

## Contributing

We welcome contributions to improve GPT Parallel. If you have ideas for improvements, bug fixes, or additional features, please feel free to open an issue or submit a pull request.

### Copyright Assignment

When submitting a pull request, you agree to assign copyright for your changes to the project maintainers. This is a standard practice in open-source projects and helps ensure the long-term viability of the project.

### Mocks and Test Cases

We are actively looking for mocks and more comprehensive test cases to help improve the reliability and stability of GPT Parallel. If you have experience in this area, your contributions will be greatly appreciated.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
