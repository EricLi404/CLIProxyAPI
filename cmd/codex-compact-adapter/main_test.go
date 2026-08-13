package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strings"
	"testing"
)

func TestEnvelopeRoundTrip(t *testing.T) {
	want := compactEnvelope{Version: "1", Kind: "checkpoint", Model: "deepseek-v4-flash", Summary: "keep the adapter state", CreatedAt: "2026-08-07T00:00:00Z"}
	blob, errEncode := encodeEnvelope(want)
	if errEncode != nil {
		t.Fatalf("encodeEnvelope() error = %v", errEncode)
	}
	got, ok := decodeEnvelope(blob)
	if !ok {
		t.Fatal("decodeEnvelope() rejected its own envelope")
	}
	if got != want {
		t.Fatalf("decoded envelope = %#v, want %#v", got, want)
	}
	if _, okForeign := decodeEnvelope("foreign-compact-state"); okForeign {
		t.Fatal("foreign compact state was accepted")
	}
}

func TestRewriteCompactionItems(t *testing.T) {
	blob, errEncode := encodeEnvelope(compactEnvelope{Version: "1", Kind: "checkpoint", Model: "deepseek-v4-flash", Summary: "checkpoint"})
	if errEncode != nil {
		t.Fatal(errEncode)
	}
	payload := map[string]any{
		"input": []any{
			map[string]any{"type": "compaction", "encrypted_content": blob},
			map[string]any{"type": "message", "role": "user", "content": "continue"},
		},
	}
	if !rewriteCompactionItems(payload) {
		t.Fatal("rewriteCompactionItems() reported no change")
	}
	input := arrayValue(payload["input"])
	if len(input) != 2 || stringValue(input[0].(map[string]any)["type"]) != "message" {
		t.Fatalf("rewritten input = %#v", input)
	}
	if !strings.Contains(stringValue(input[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]), "checkpoint") {
		t.Fatalf("checkpoint was not restored: %#v", input[0])
	}
}

func TestLocalCompactionV2AndReplay(t *testing.T) {
	var summaryRequest map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if r.URL.Path != "/v1/responses" {
			t.Errorf("upstream path = %q", r.URL.Path)
		}
		if errDecode := json.Unmarshal(body, &summaryRequest); errDecode != nil {
			t.Errorf("summary request JSON: %v", errDecode)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp_summary","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"checkpoint summary"}]}]}`)
	}))
	defer upstream.Close()
	target, _ := url.Parse(upstream.URL)
	handler := newAdapterHandler(adapterConfig{TargetURL: target, Models: mustPatterns(t, "deepseek-*")})
	requestBody := `{"model":"deepseek-v4-flash","stream":true,"input":[{"type":"message","role":"user","content":"history"},{"type":"compaction_trigger"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(requestBody))
	req.Header.Set("Authorization", "Bearer test")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), `"type":"compaction"`) || !strings.Contains(recorder.Body.String(), "response.completed") {
		t.Fatalf("compact response = status %d body %s", recorder.Code, recorder.Body.String())
	}
	if hasItemType(arrayValue(summaryRequest["input"]), "compaction_trigger") {
		t.Fatal("compaction_trigger reached the summary upstream")
	}

	var response map[string]any
	if errDecode := json.Unmarshal([]byte(`{"model":"deepseek-v4-flash","input":[{"type":"compaction","encrypted_content":""}]}`), &response); errDecode != nil {
		t.Fatal(errDecode)
	}
	for _, raw := range strings.Split(recorder.Body.String(), "data: ") {
		if strings.Contains(raw, `"type":"compaction"`) {
			var event map[string]any
			if json.Unmarshal([]byte(strings.SplitN(raw, "\n", 2)[0]), &event) == nil {
				if item, ok := event["item"].(map[string]any); ok {
					response["input"] = []any{item}
				}
			}
		}
	}
	if !rewriteCompactionItems(response) {
		t.Fatal("replay did not rewrite adapter compaction state")
	}
}

func TestGPTPassesThrough(t *testing.T) {
	var gotBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"normal","output":[{"type":"message"}]}`)
	}))
	defer upstream.Close()
	target, _ := url.Parse(upstream.URL)
	handler := newAdapterHandler(adapterConfig{TargetURL: target, Models: mustPatterns(t, "deepseek-*")})
	body := `{"model":"gpt-5.6-sol","input":[{"type":"compaction_trigger"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusOK || string(gotBody) != body {
		t.Fatalf("GPT request was modified: status=%d body=%s upstream=%s", recorder.Code, recorder.Body.String(), gotBody)
	}
}

func mustPatterns(t *testing.T, values ...string) []*regexp.Regexp {
	t.Helper()
	result := make([]*regexp.Regexp, 0, len(values))
	for _, value := range values {
		expression, errPattern := wildcardRegexp(value)
		if errPattern != nil {
			t.Fatal(errPattern)
		}
		result = append(result, expression)
	}
	return result
}
