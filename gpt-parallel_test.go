package gptparallel_test

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	gptparallel "github.com/tbiehn/gptparallel"

	backoff "github.com/cenkalti/backoff/v4"
	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/vbauerster/mpb/v8"
)

func TestNewGPTParallel(t *testing.T) {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	progress := mpb.New()
	backoffSettings := backoff.NewExponentialBackOff()

	gptParallel := gptparallel.NewGPTParallel(context.Background(), client, progress, backoffSettings, nil)

	assert.NotNil(t, gptParallel, "GPTParallel instance should not be nil")
	assert.NotNil(t, gptParallel.Client, "Client should not be nil")
	assert.NotNil(t, gptParallel.Progress, "Progress should not be nil")
	assert.NotNil(t, gptParallel.BackoffSettings, "BackoffSettings should not be nil")
	assert.NotNil(t, gptParallel.Logger, "Logger should not be nil")
}

func TestRunRequests(t *testing.T) {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	ctx := context.Background()

	gptParallel := gptparallel.NewGPTParallel(ctx, client, mpb.New(), backoff.NewExponentialBackOff(), nil)

	requests := []gptparallel.RequestWithCallback{
		{
			Request: openai.ChatCompletionRequest{
				Messages: []openai.ChatCompletionMessage{
					{
						Role:    "system",
						Content: "You are a helpful assistant.",
					},
					{
						Role:    "user",
						Content: "Who won the world series in 2020?",
					},
				},
				Model:     "gpt-3.5-turbo",
				MaxTokens: 10,
			},
			Callback: func(result gptparallel.RequestResult) {
				if result.Err != nil {
					t.Errorf("Request failed with error: %v", result.Err)
				} else {
					fmt.Printf("Received response: %s\n", result.Response)
				}
			},
		},
	}

	concurrency := 1
	timeout := 10 * time.Second

	done := make(chan struct{})
	go func() {
		gptParallel.RunRequests(requests, concurrency)
		close(done)
	}()

	select {
	case <-done:
		// Completed successfully
	case <-time.After(timeout):
		t.Errorf("Test timed out after %v", timeout)
	}

}

func TestRunRequestsMPBNil(t *testing.T) {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	ctx := context.Background()

	gptParallel := gptparallel.NewGPTParallel(ctx, client, nil, backoff.NewExponentialBackOff(), nil)

	requests := []gptparallel.RequestWithCallback{
		{
			Request: openai.ChatCompletionRequest{
				Messages: []openai.ChatCompletionMessage{
					{
						Role:    "system",
						Content: "You are a helpful assistant.",
					},
					{
						Role:    "user",
						Content: "Who won the world series in 2020?",
					},
				},
				Model:     "gpt-3.5-turbo",
				MaxTokens: 10,
			},
			Callback: func(result gptparallel.RequestResult) {
				if result.Err != nil {
					t.Errorf("Request failed with error: %v", result.Err)
				} else {
					fmt.Printf("Received response: %s\n", result.Response)
				}
			},
		},
	}

	concurrency := 1
	timeout := 10 * time.Second

	done := make(chan struct{})
	go func() {
		gptParallel.RunRequests(requests, concurrency)
		close(done)
	}()

	select {
	case <-done:
		// Completed successfully
	case <-time.After(timeout):
		t.Errorf("Test timed out after %v", timeout)
	}

}
