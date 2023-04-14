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

func ExampleRunRequestsChan() {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	ctx := context.Background()

	backoffSettings := backoff.NewExponentialBackOff()

	g := gptparallel.NewGPTParallel(ctx, client, nil, backoffSettings, nil)

	requestsChan := make(chan gptparallel.RequestWithCallback)

	go func() {
		for i := 0; i < 5; i++ {
			request := openai.ChatCompletionRequest{
				Model:       "gpt-3.5-turbo",
				Messages:    []openai.ChatCompletionMessage{{Role: "system", Content: "You are a helpful assistant."}, {Role: "user", Content: fmt.Sprintf("What is %d * %d?", i+1, i+2)}},
				MaxTokens:   10,
				Temperature: 0,
			}
			requestsChan <- gptparallel.RequestWithCallback{
				Request: request,
				Callback: func(result gptparallel.RequestResult) {
					if result.Err != nil {
						fmt.Printf("Request failed: %v\n", result.Err)
					} else {
						fmt.Printf("Result:\n")
					}
				},
			}
		}
		close(requestsChan)
	}()

	resultsChan := g.RunRequestsChan(requestsChan, 2)

	for result := range resultsChan {
		if result.Err != nil {
			fmt.Printf("Request failed: %v\n", result.Err)
		} else {
			fmt.Print("Result:\n") // elide result to satisfy, result.Response)
		}
	}
	// We've made the test dumb to account for un-mocked openai.
	// Unordered output:
	// Result:
	// Result:
	// Result:
	// Result:
	// Result:

}
