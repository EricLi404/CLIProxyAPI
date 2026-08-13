package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const envelopePrefix = "ccmp1."

type adapterConfig struct {
	TargetURL   *url.URL
	Models      []*regexp.Regexp
	MaxBodySize int64
}

type compactEnvelope struct {
	Version   string `json:"version"`
	Kind      string `json:"kind"`
	Model     string `json:"model"`
	Summary   string `json:"summary"`
	CreatedAt string `json:"created_at"`
}

func main() {
	target := getenv("CODEX_COMPACT_TARGET", "http://127.0.0.1:8317")
	targetURL, errParse := url.Parse(target)
	if errParse != nil || targetURL.Scheme == "" || targetURL.Host == "" {
		log.Fatalf("invalid CODEX_COMPACT_TARGET: %q", target)
	}
	patterns := splitCSV(getenv("CODEX_COMPACT_MODELS", "deepseek-*,glm-*,mimo-*,minimax-*"))
	modelRegexps := make([]*regexp.Regexp, 0, len(patterns))
	for _, pattern := range patterns {
		expr, errPattern := wildcardRegexp(pattern)
		if errPattern != nil {
			log.Fatalf("invalid compact model pattern %q: %v", pattern, errPattern)
		}
		modelRegexps = append(modelRegexps, expr)
	}
	server := &http.Server{
		Addr:              getenv("CODEX_COMPACT_LISTEN", "127.0.0.1:8320"),
		Handler:           newAdapterHandler(adapterConfig{TargetURL: targetURL, Models: modelRegexps, MaxBodySize: 64 << 20}),
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Printf("codex compact adapter listening on %s, target=%s, models=%s", server.Addr, targetURL, strings.Join(patterns, ","))
	if errServe := server.ListenAndServe(); errServe != nil && !errors.Is(errServe, http.ErrServerClosed) {
		log.Fatal(errServe)
	}
}

func newAdapterHandler(cfg adapterConfig) http.Handler {
	if cfg.MaxBodySize <= 0 {
		cfg.MaxBodySize = 64 << 20
	}
	client := &http.Client{}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"ok":true,"service":"codex-compact-adapter"}`)
			return
		}
		if r.Method != http.MethodPost {
			proxyRequest(w, r, cfg, client, nil)
			return
		}

		body, errRead := io.ReadAll(io.LimitReader(r.Body, cfg.MaxBodySize+1))
		if errRead != nil {
			http.Error(w, "failed to read request body", http.StatusBadRequest)
			return
		}
		if int64(len(body)) > cfg.MaxBodySize {
			http.Error(w, "request body too large", http.StatusRequestEntityTooLarge)
			return
		}
		var payload map[string]any
		if errDecode := json.Unmarshal(body, &payload); errDecode != nil {
			proxyRequest(w, r, cfg, client, body)
			return
		}

		model := stringValue(payload["model"])
		input := arrayValue(payload["input"])
		isV1 := strings.HasSuffix(strings.TrimSuffix(r.URL.Path, "/"), "/responses/compact")
		isV2 := hasItemType(input, "compaction_trigger")
		modelMatched := cfg.modelMatches(model)
		log.Printf("request method=%s path=%s model=%q input_types=%v is_v1=%t is_v2=%t model_matched=%t", r.Method, r.URL.Path, model, itemTypes(input), isV1, isV2, modelMatched)
		if modelMatched && (isV1 || isV2) {
			serveLocalCompaction(w, r, cfg, client, payload, model, isV1)
			return
		}

		if rewriteCompactionItems(payload) {
			body, _ = json.Marshal(payload)
		}
		proxyRequest(w, r, cfg, client, body)
	})
}

func serveLocalCompaction(w http.ResponseWriter, r *http.Request, cfg adapterConfig, client *http.Client, payload map[string]any, model string, v1 bool) {
	input := arrayValue(payload["input"])
	cleanInput := removeItemTypes(input, "compaction_trigger")
	summaryPayload := cloneMap(payload)
	summaryPayload["input"] = append(cleanInput, map[string]any{
		"type": "message",
		"role": "user",
		"content": []any{map[string]any{
			"type": "input_text",
			"text": compactPrompt,
		}},
	})
	summaryPayload["stream"] = false
	summaryPayload["store"] = false
	delete(summaryPayload, "previous_response_id")
	delete(summaryPayload, "tools")
	delete(summaryPayload, "tool_choice")
	// A repeated compaction must replay our own checkpoint as ordinary context
	// before asking the upstream model to produce the next checkpoint.
	rewriteCompactionItems(summaryPayload)

	summary, errSummary := requestSummary(r.Context(), cfg, client, r, summaryPayload)
	if errSummary != nil {
		log.Printf("local compact summary failed for model=%s: %v; using bounded fallback", model, errSummary)
		summary = fallbackSummary(cleanInput)
	}
	blob, errEnvelope := encodeEnvelope(compactEnvelope{
		Version:   "1",
		Kind:      "checkpoint",
		Model:     model,
		Summary:   summary,
		CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
	})
	if errEnvelope != nil {
		http.Error(w, "failed to encode compact state", http.StatusInternalServerError)
		return
	}
	responseID := "resp_compact_" + strconv.FormatInt(time.Now().UnixNano(), 10)
	item := map[string]any{
		"id":                "cmp_" + strings.TrimPrefix(responseID, "resp_"),
		"type":              "compaction",
		"status":            "completed",
		"encrypted_content": blob,
	}
	if v1 || !wantsStream(payload) {
		writeJSON(w, http.StatusOK, map[string]any{
			"id":     responseID,
			"object": "response.compaction",
			"model":  model,
			"status": "completed",
			"output": []any{item},
			"usage":  map[string]any{"input_tokens": 0, "output_tokens": estimateTokens(summary), "total_tokens": estimateTokens(summary)},
		})
		return
	}
	writeCompactionSSE(w, responseID, model, item)
}

const compactPrompt = `Create a durable checkpoint summary for continuing this coding session. Return only the summary text. Preserve the user's goal, current progress, important decisions, constraints, file paths, commands, tool results, errors, unresolved questions, and the exact next actions needed. Do not invent facts and do not include conversational filler.`

func requestSummary(ctx context.Context, cfg adapterConfig, client *http.Client, original *http.Request, payload map[string]any) (string, error) {
	body, errMarshal := json.Marshal(payload)
	if errMarshal != nil {
		return "", errMarshal
	}
	endpoint := *cfg.TargetURL
	endpoint.Path = joinURLPath(endpoint.Path, "/v1/responses")
	req, errRequest := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), bytes.NewReader(body))
	if errRequest != nil {
		return "", errRequest
	}
	copyRequestHeaders(req.Header, original.Header)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Del("Content-Length")
	resp, errDo := client.Do(req)
	if errDo != nil {
		return "", errDo
	}
	defer resp.Body.Close()
	data, errRead := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if errRead != nil {
		return "", errRead
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("upstream status %d: %s", resp.StatusCode, summarizeBody(data))
	}
	return extractSummary(data), nil
}

func proxyRequest(w http.ResponseWriter, r *http.Request, cfg adapterConfig, client *http.Client, body []byte) {
	target := *cfg.TargetURL
	target.Path = joinURLPath(target.Path, r.URL.Path)
	target.RawQuery = r.URL.RawQuery
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req, errRequest := http.NewRequestWithContext(r.Context(), r.Method, target.String(), reader)
	if errRequest != nil {
		http.Error(w, "failed to build upstream request", http.StatusBadGateway)
		return
	}
	copyRequestHeaders(req.Header, r.Header)
	if body != nil {
		req.Header.Set("Content-Length", strconv.Itoa(len(body)))
	}
	resp, errDo := client.Do(req)
	if errDo != nil {
		http.Error(w, "upstream request failed", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	copyResponseHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func writeCompactionSSE(w http.ResponseWriter, responseID, model string, item map[string]any) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, _ := w.(http.Flusher)
	writeEvent := func(event string, payload map[string]any) {
		data, _ := json.Marshal(payload)
		_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
		if flusher != nil {
			flusher.Flush()
		}
	}
	base := map[string]any{"id": responseID, "object": "response", "model": model, "status": "in_progress", "output": []any{}}
	writeEvent("response.created", map[string]any{"type": "response.created", "response": base})
	writeEvent("response.in_progress", map[string]any{"type": "response.in_progress", "response": base})
	writeEvent("response.output_item.added", map[string]any{"type": "response.output_item.added", "output_index": 0, "item": item})
	writeEvent("response.output_item.done", map[string]any{"type": "response.output_item.done", "output_index": 0, "item": item})
	completed := cloneMap(base)
	completed["status"] = "completed"
	completed["output"] = []any{item}
	writeEvent("response.completed", map[string]any{"type": "response.completed", "response": completed})
}

func rewriteCompactionItems(payload map[string]any) bool {
	input, ok := payload["input"].([]any)
	if !ok {
		return false
	}
	changed := false
	rewritten := make([]any, 0, len(input))
	for _, raw := range input {
		item, okItem := raw.(map[string]any)
		if !okItem {
			rewritten = append(rewritten, raw)
			continue
		}
		typeName := stringValue(item["type"])
		if typeName != "compaction" && typeName != "context_compaction" {
			rewritten = append(rewritten, raw)
			continue
		}
		blob := stringValue(item["encrypted_content"])
		envelope, okEnvelope := decodeEnvelope(blob)
		if !okEnvelope {
			changed = true
			continue
		}
		changed = true
		rewritten = append(rewritten, map[string]any{
			"type": "message",
			"role": "user",
			"content": []any{map[string]any{
				"type": "input_text",
				"text": "[Conversation checkpoint]\n" + envelope.Summary,
			}},
		})
	}
	if changed {
		payload["input"] = rewritten
	}
	return changed
}

func encodeEnvelope(envelope compactEnvelope) (string, error) {
	data, errMarshal := json.Marshal(envelope)
	if errMarshal != nil {
		return "", errMarshal
	}
	return envelopePrefix + base64.RawURLEncoding.EncodeToString(data), nil
}

func decodeEnvelope(blob string) (compactEnvelope, bool) {
	if !strings.HasPrefix(blob, envelopePrefix) {
		return compactEnvelope{}, false
	}
	data, errDecode := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(blob, envelopePrefix))
	if errDecode != nil {
		return compactEnvelope{}, false
	}
	var envelope compactEnvelope
	if json.Unmarshal(data, &envelope) != nil || envelope.Version != "1" || envelope.Kind != "checkpoint" || strings.TrimSpace(envelope.Summary) == "" {
		return compactEnvelope{}, false
	}
	return envelope, true
}

func extractSummary(data []byte) string {
	var payload map[string]any
	if json.Unmarshal(data, &payload) != nil {
		return ""
	}
	if output, ok := payload["output"].([]any); ok {
		for i := len(output) - 1; i >= 0; i-- {
			if text := extractText(output[i]); strings.TrimSpace(text) != "" {
				return strings.TrimSpace(text)
			}
		}
	}
	if choices, ok := payload["choices"].([]any); ok {
		for _, choice := range choices {
			if item, okItem := choice.(map[string]any); okItem {
				if text := extractText(item["message"]); strings.TrimSpace(text) != "" {
					return strings.TrimSpace(text)
				}
			}
		}
	}
	return ""
}

func extractText(value any) string {
	item, ok := value.(map[string]any)
	if !ok {
		return stringValue(value)
	}
	if text := stringValue(item["output_text"]); text != "" {
		return text
	}
	if text := stringValue(item["text"]); text != "" {
		return text
	}
	if content, okContent := item["content"].([]any); okContent {
		var parts []string
		for _, part := range content {
			if text := extractText(part); strings.TrimSpace(text) != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

func fallbackSummary(input []any) string {
	const maxBytes = 24000
	data, _ := json.Marshal(input)
	if len(data) > maxBytes {
		data = data[len(data)-maxBytes:]
	}
	return "A compact summary could not be generated. The most recent retained context is:\n" + string(data)
}

func estimateTokens(text string) int { return max(1, len([]rune(text))/4) }

func hasItemType(input []any, wanted string) bool {
	for _, raw := range input {
		if item, ok := raw.(map[string]any); ok && stringValue(item["type"]) == wanted {
			return true
		}
	}
	return false
}

func itemTypes(input []any) []string {
	types := make([]string, 0, len(input))
	for _, raw := range input {
		if item, ok := raw.(map[string]any); ok {
			types = append(types, stringValue(item["type"]))
		}
	}
	return types
}

func removeItemTypes(input []any, types ...string) []any {
	set := make(map[string]struct{}, len(types))
	for _, typeName := range types {
		set[typeName] = struct{}{}
	}
	out := make([]any, 0, len(input))
	for _, raw := range input {
		item, ok := raw.(map[string]any)
		if ok {
			if _, found := set[stringValue(item["type"])]; found {
				continue
			}
		}
		out = append(out, raw)
	}
	return out
}

func arrayValue(value any) []any {
	items, _ := value.([]any)
	return items
}

func stringValue(value any) string {
	text, _ := value.(string)
	return text
}

func cloneMap(input map[string]any) map[string]any {
	output := make(map[string]any, len(input))
	for key, value := range input {
		output[key] = value
	}
	return output
}

func wantsStream(payload map[string]any) bool {
	stream, ok := payload["stream"].(bool)
	return ok && stream
}

func (cfg adapterConfig) modelMatches(model string) bool {
	for _, expression := range cfg.Models {
		if expression.MatchString(model) {
			return true
		}
	}
	return false
}

func wildcardRegexp(pattern string) (*regexp.Regexp, error) {
	quoted := regexp.QuoteMeta(strings.TrimSpace(pattern))
	quoted = strings.ReplaceAll(quoted, `\*`, `.*`)
	return regexp.Compile(`^(?i:` + quoted + `)$`)
}

func splitCSV(value string) []string {
	var result []string
	for _, item := range strings.Split(value, ",") {
		if trimmed := strings.TrimSpace(item); trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}

func joinURLPath(basePath, requestPath string) string {
	if basePath == "" {
		return requestPath
	}
	return path.Join(basePath, requestPath)
}

func copyRequestHeaders(dst, src http.Header) {
	for key, values := range src {
		if isHopByHopHeader(key) {
			continue
		}
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func copyResponseHeaders(dst, src http.Header) {
	for key, values := range src {
		if isHopByHopHeader(key) {
			continue
		}
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func isHopByHopHeader(key string) bool {
	switch strings.ToLower(key) {
	case "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade":
		return true
	default:
		return false
	}
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func summarizeBody(body []byte) string {
	text := strings.TrimSpace(string(body))
	if len(text) > 500 {
		return text[:500]
	}
	return text
}

func getenv(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
