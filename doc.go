// gptparallel project doc.go

/*
Package gptparallel provides an efficient way to handle multiple requests to OpenAI GPT models using the go-openai library. It simplifies the execution of parallel requests and incorporates retry logic.

GPTParallel wraps the go-openai library and manages concurrent requests to the OpenAI API or the Azure OpenAI API. It allows users to set a configurable number of concurrent requests and a custom exponential backoff strategy to handle retries in case of failures.

This package is designed to efficiently manage concurrent requests to GPT models while providing progress updates and handling retries with exponential backoff strategy.
*/
package gptparallel
