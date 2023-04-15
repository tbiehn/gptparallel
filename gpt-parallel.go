package gptparallel

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/pkoukk/tiktoken-go"

	backoff "github.com/cenkalti/backoff/v4"
	openai "github.com/sashabaranov/go-openai"
	"github.com/vbauerster/mpb/v8"
	"github.com/vbauerster/mpb/v8/decor"
)

// RequestWithCallback: A struct containing an openai.ChatCompletionRequest and a callback function to process the request result.

type RequestWithCallback struct {
	Request    openai.ChatCompletionRequest
	Callback   func(result RequestResult)
	Identifier string
}

// RequestResult: A struct containing the original request, the response, the finish reason, and any errors that occurred during the request.
type RequestResult struct {
	Request      openai.ChatCompletionRequest `json:"request"`
	Response     string                       `json:"response"`
	Identifier   string                       `json:"identifier"`
	FinishReason string                       `json:"finish_reason"`
	Err          error                        `json:"error,omitempty"`
}

// GPTParallel: The main struct responsible for managing concurrent requests, progress bars, and backoff settings.
type GPTParallel struct {
	ctx             context.Context
	Client          *openai.Client
	Progress        *mpb.Progress
	BackoffSettings *backoff.ExponentialBackOff
	Logger          Logger
}

// NewGPTParallel: A function that creates a new GPTParallel instance with the given context, client, progress, backoff settings, and optional logger.
func NewGPTParallel(context context.Context, client *openai.Client, progress *mpb.Progress, backoffSettings *backoff.ExponentialBackOff, optLogger Logger) *GPTParallel {
	var logger Logger
	if optLogger != nil {
		logger = optLogger
	} else {
		logger = &noOpLogger{}
	}
	return &GPTParallel{ctx: context, Client: client, Progress: progress, BackoffSettings: backoffSettings, Logger: logger}
}

// RunRequests: A method that executes all requests in parallel with the given concurrency level and manages progress bars and retries.
func (g *GPTParallel) RunRequests(requests []RequestWithCallback, concurrency int) {
	if concurrency <= 0 {
		concurrency = 1
	}

	// Calculate the total number of tokens in all requests
	totalTokens := int64(0)
	for _, reqWithCallback := range requests {
		totalTokens += int64(reqWithCallback.Request.MaxTokens)
	}

	// Add a high priority progress bar to track overall completion
	var overallBar *mpb.Bar
	if g.Progress != nil {
		overallBar = g.Progress.AddBar(totalTokens,
			mpb.BarPriority(-1),
			mpb.PrependDecorators(
				decor.Name("Overall: "),
				decor.CountersNoUnit(" (%d/%d)"),
			),
			mpb.AppendDecorators(
				decor.OnComplete(
					decor.AverageETA(decor.ET_STYLE_GO, decor.WCSyncWidth), "completed",
				),
			),
		)
	}

	requestsChan := make(chan RequestWithCallback)
	wg := sync.WaitGroup{}

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		barName := fmt.Sprintf("Query # %d", i)
		go func() {
			defer wg.Done()
			for reqWithCallback := range requestsChan {
				result, finish, err := g.chatCompletionWithExponentialBackoff(barName, reqWithCallback.Request)
				reqWithCallback.Callback(RequestResult{
					Request:      reqWithCallback.Request,
					FinishReason: finish,
					Identifier:   reqWithCallback.Identifier,
					Response:     result,
					Err:          err,
				})
				if overallBar != nil {
					overallBar.IncrBy(reqWithCallback.Request.MaxTokens)
				}

			}
		}()
	}

	for _, reqWithCallback := range requests {
		requestsChan <- reqWithCallback
	}

	wg.Wait()

	close(requestsChan)

	if overallBar != nil {
		overallBar.Abort(true)
	}

}

// RunRequestsChan: A method that executes requests received from a channel in parallel with the given concurrency level, manages progress bars and retries, and sends results to a channel.
func (g *GPTParallel) RunRequestsChan(requestsChan <-chan RequestWithCallback, concurrency int) <-chan RequestResult {
	resultsChan := make(chan RequestResult)

	if concurrency <= 0 {
		concurrency = 1
	}

	// Calculate the total number of tokens in all requests
	var totalTokens int64

	var overallBar *mpb.Bar
	if g.Progress != nil {
		overallBar = g.Progress.AddBar(totalTokens,
			mpb.BarPriority(-1),
			mpb.PrependDecorators(
				decor.Name("Overall: "),
				decor.CountersNoUnit(" (%d/%d)"),
			),
			mpb.AppendDecorators(
				decor.OnComplete(
					decor.AverageETA(decor.ET_STYLE_GO, decor.WCSyncWidth), "completed",
				),
			),
		)
	}

	wg := sync.WaitGroup{}

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		barName := fmt.Sprintf("Query # %d", i)
		go func() {
			defer wg.Done()
			for reqWithCallback := range requestsChan {
				totalTokens += int64(reqWithCallback.Request.MaxTokens)
				if overallBar != nil {
					overallBar.SetTotal(totalTokens, false)
				}

				result, finish, err := g.chatCompletionWithExponentialBackoff(barName, reqWithCallback.Request)
				go reqWithCallback.Callback(RequestResult{
					Request:      reqWithCallback.Request,
					FinishReason: finish,
					Response:     result,
					Identifier:   reqWithCallback.Identifier,
					Err:          err,
				})

				resultsChan <- RequestResult{
					Request:      reqWithCallback.Request,
					FinishReason: finish,
					Response:     result,
					Identifier:   reqWithCallback.Identifier,
					Err:          err,
				}
				if overallBar != nil {
					overallBar.IncrBy(reqWithCallback.Request.MaxTokens)
				}
			}
		}()
	}

	go func() {
		wg.Wait()
		if overallBar != nil {
			overallBar.Abort(true)
		}
		close(resultsChan)
	}()

	return resultsChan
}

func (g *GPTParallel) chatCompletionWithBackoff(req openai.ChatCompletionRequest, bar *mpb.Bar) (string, string, error) {
	req.Stream = true
	stream, err := g.Client.CreateChatCompletionStream(g.ctx, req)
	if err != nil {
		return "", "", err
	}
	defer stream.Close()

	// Get encoding
	model := req.Model
	if strings.HasPrefix(model, "gpt-4") {
		model = openai.GPT4
	} else if strings.HasPrefix(model, "gpt-3.5-turbo") {
		model = openai.GPT3Dot5Turbo
	}

	/*if rand.Float64() < .5 {
		return "", "", fmt.Errorf("Just a synthetic error.")
	}*/

	encoding, err := tiktoken.EncodingForModel(model)
	if err != nil {
		return "", "", err
	}

	result := ""
	lastFinish := ""

	for {
		startTime := time.Now()
		response, err := stream.Recv()
		iterDuration := time.Since(startTime)

		if errors.Is(err, io.EOF) {
			if bar != nil {
				//bar.SetTotal(int64(req.MaxTokens), true)
				//bar.Abort(false)
				bar.EwmaIncrInt64(int64(req.MaxTokens)-bar.Current(), iterDuration)
				bar.Wait()
			}
			return result, lastFinish, nil
		}

		if err != nil {

			return result, "", err
		}

		if len(response.Choices) > 0 {
			chunk := response.Choices[0].Delta.Content
			result += chunk
			lastFinish = response.Choices[0].FinishReason
			//g.Logger.Debug("Finish reason", response.Choices[0].FinishReason)

			if bar != nil {
				tokenCount := len(encoding.Encode(chunk, nil, nil))
				bar.EwmaIncrInt64(int64(tokenCount), iterDuration)
			}
		}
	}
}

func (g *GPTParallel) chatCompletionWithExponentialBackoff(name string, req openai.ChatCompletionRequest) (string, string, error) {
	var result, finish string

	//backoff.Retry contract only permits returning an error.
	operation := func() error {
		var bar *mpb.Bar
		if g.Progress != nil {
			g.Logger.Debugf("Adding a bar.")
			bar = g.Progress.AddBar(int64(req.MaxTokens),
				mpb.PrependDecorators(
					decor.Name(name),
					//decor.CountersNoUnit(" (%d/%d)"),
					decor.AverageSpeed(0, "(%.1f TPS)"),
				),
				mpb.AppendDecorators(
					decor.OnComplete(
						decor.EwmaETA(decor.ET_STYLE_GO, 60, decor.WCSyncWidth), "done",
					),
				),
				mpb.BarRemoveOnComplete(),
			)
		}

		resultx, finishx, err := g.chatCompletionWithBackoff(req, bar)
		g.Logger.Debug("Got this response;", resultx, finishx, err)

		if err != nil {
			g.Logger.Debug("Have an error.", err != nil)
			if bar != nil {
				//TODO: Extract bar to parent, manage retry status out there.
				//bar.SetRefill(int64(0))
				bar.Abort(true)
				bar.Wait()
			}
			return err
		}
		result = resultx
		finish = finishx
		return nil
	}

	ctx, cancel := context.WithCancel(g.ctx)

	backoffContext := backoff.WithContext(g.BackoffSettings, ctx)

	notify := func(err error, duration time.Duration) {
		barName := name + " Retry"

		g.Logger.Debugf("Error: %T %+v", err, err)
		var apiErr *openai.APIError
		if errors.As(err, &apiErr) {
			g.Logger.Debugf("Received an APIError: %d, %s, %s, %s", apiErr.StatusCode, apiErr.Type, apiErr)
			//apiErr.StatusCode is 0 when Streaming.
			switch apiErr.Type {
			case "invalid_request_error":
				g.Logger.Error("Invalid Request Error")
				cancel()
				break
			}
			barName = barName + " " + apiErr.Type
		}
		var reqErr *openai.RequestError
		if errors.As(err, &reqErr) {
			g.Logger.Errorf("Received RequestError: %s, %d", reqErr.Err, reqErr.StatusCode)
		}

		//This error doesn't follow the standard.
		if strings.Contains(err.Error(), "You didn't provide an API key.") {
			g.Logger.Error("No API Key was provided.")
			cancel()
		}

		//We are backed off for time.Duration.
		//Render a 'retrying bar' that completes with duration is up.
		if g.Progress != nil {

			retryBar := g.Progress.AddBar(int64(duration),
				mpb.PrependDecorators(
					decor.Name(barName),
					//decor.CountersNoUnit(" (%d/%d)"),
				),
				mpb.AppendDecorators(
					decor.OnComplete(
						decor.AverageETA(decor.ET_STYLE_GO, decor.WCSyncWidth), "retrying",
					),
				),
				mpb.BarRemoveOnComplete(),
			)

			ticker := time.NewTicker(10 * time.Millisecond)
			startTime := time.Now()
			for {
				select {
				case <-g.ctx.Done():
					ticker.Stop()
					retryBar.Abort(false)
					retryBar.Wait()
					return
				case <-ticker.C:
					elapsed := time.Since(startTime)
					if elapsed >= duration {
						ticker.Stop()
						retryBar.SetCurrent(int64(duration))
						return
					}
					retryBar.SetCurrent(int64(elapsed))
				}
			}
		}

	}

	err := backoff.RetryNotify(operation, backoffContext, notify)
	return result, finish, err
}

// We've omitted 'Fatal' errors. This library shouldn't cause any panics or os.Exit()s.

// Logger: An interface to support different logging implementations, with a default no-op Logger provided.
type Logger interface {
	Debug(args ...interface{})
	Debugf(format string, args ...interface{})
	Info(args ...interface{})
	Infof(format string, args ...interface{})
	Warn(args ...interface{})
	Warnf(format string, args ...interface{})
	Error(args ...interface{})
	Errorf(format string, args ...interface{})
}

// noOpLogger: A no-operation logger implementation that does not log anything. This is the default logger used if no custom logger is provided.
type noOpLogger struct{}

func (n *noOpLogger) Debug(args ...interface{})                 {}
func (n *noOpLogger) Debugf(format string, args ...interface{}) {}
func (n *noOpLogger) Info(args ...interface{})                  {}
func (n *noOpLogger) Infof(format string, args ...interface{})  {}
func (n *noOpLogger) Warn(args ...interface{})                  {}
func (n *noOpLogger) Warnf(format string, args ...interface{})  {}
func (n *noOpLogger) Error(args ...interface{})                 {}
func (n *noOpLogger) Errorf(format string, args ...interface{}) {}
