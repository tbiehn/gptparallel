package gptparallel_test

import (
	"context"
	"fmt"
	"os"

	gptparallel "github.com/tbiehn/gptparallel"

	backoff "github.com/cenkalti/backoff/v4"
	openai "github.com/sashabaranov/go-openai"
)

func Example() {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	ctx := context.Background()

	backoffSettings := backoff.NewExponentialBackOff()

	gptParallel := gptparallel.NewGPTParallel(ctx, client, nil, backoffSettings, nil)

	// Prepare requests and their respective callbacks
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
					fmt.Printf("Request failed with error: %v", result.Err)
				} else {
					fmt.Print("Received response!\n")
				}
			},
		},
		// More requests can be added here
	}

	concurrency := 2 // Number of concurrent requests
	gptParallel.RunRequests(requests, concurrency)
	// Output:
	// Received response!
}
