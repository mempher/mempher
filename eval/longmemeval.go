// The LongMemEval loader.

package eval

import (
	"encoding/json"
	"fmt"
	"iter"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/mempher/mempher"
)

// longMemEvalDate is how the benchmark writes an instant: "2023/05/30 (Tue) 23:40".
const longMemEvalDate = "2006/01/02 (Mon) 15:04"

// LongMemEval reads the LongMemEval benchmark at path, which is the
// longmemeval_s, _m or _oracle file from the dataset, unmodified.
//
// It labels at turn level rather than session level. The benchmark marks the
// individual turns that carry an answer, and an episode in mempher is a turn, so
// the question being scored is the exact one a caller cares about: did recall
// return the message that holds the answer, not merely the conversation it was
// somewhere inside.
//
// The file is read one question at a time. longmemeval_s is 278MB and decodes
// to several times that, which is worth avoiding for a number that is computed
// in one pass anyway.
//
// Questions labelling no relevant turn -- the abstention cases, where the right
// answer is that the memory holds nothing -- are yielded with an empty Relevant
// and skipped by [Run]. Retrieval has no target to hit there, and any score for
// it would be arbitrary.
func LongMemEval(path string) Dataset {
	return Dataset{
		Name:      "LongMemEval (" + filepath.Base(path) + ")",
		Questions: longMemEvalQuestions(path),
	}
}

func longMemEvalQuestions(path string) iter.Seq2[Question, error] {
	return func(yield func(Question, error) bool) {
		file, err := os.Open(path)
		if err != nil {
			yield(Question{}, fmt.Errorf("eval: open the benchmark: %w", err))
			return
		}
		defer func() { _ = file.Close() }()

		decoder := json.NewDecoder(file)
		// The opening bracket of the top-level array, consumed so that Decode
		// below reads one element rather than the whole file.
		if _, err := decoder.Token(); err != nil {
			yield(Question{}, fmt.Errorf("eval: read the benchmark: %w", err))
			return
		}
		for decoder.More() {
			var instance lmeInstance
			if err := decoder.Decode(&instance); err != nil {
				yield(Question{}, fmt.Errorf("eval: decode a question: %w", err))
				return
			}
			if !yield(instance.question()) {
				return
			}
		}
	}
}

type lmeInstance struct {
	QuestionID         string      `json:"question_id"`
	QuestionType       string      `json:"question_type"`
	Question           string      `json:"question"`
	Answer             loose       `json:"answer"`
	QuestionDate       string      `json:"question_date"`
	HaystackDates      []string    `json:"haystack_dates"`
	HaystackSessionIDs []string    `json:"haystack_session_ids"`
	HaystackSessions   [][]lmeTurn `json:"haystack_sessions"`
}

type lmeTurn struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	HasAnswer bool   `json:"has_answer"`
}

func (in lmeInstance) question() (Question, error) {
	asked, err := time.Parse(longMemEvalDate, in.QuestionDate)
	if err != nil {
		return Question{}, fmt.Errorf("eval: %s: question date %q: %w",
			in.QuestionID, in.QuestionDate, err)
	}

	q := Question{
		ID:     in.QuestionID,
		Type:   in.QuestionType,
		Query:  in.Question,
		Answer: string(in.Answer),
		Asked:  asked,
	}
	for i, session := range in.HaystackSessions {
		if i >= len(in.HaystackDates) {
			return Question{}, fmt.Errorf("eval: %s: session %d has no date", in.QuestionID, i)
		}
		occurred, err := time.Parse(longMemEvalDate, in.HaystackDates[i])
		if err != nil {
			return Question{}, fmt.Errorf("eval: %s: session %d date %q: %w",
				in.QuestionID, i, in.HaystackDates[i], err)
		}
		sessionID := ""
		if i < len(in.HaystackSessionIDs) {
			sessionID = in.HaystackSessionIDs[i]
		}

		for _, turn := range session {
			// A handful of turns in the published file are blank. An episode
			// cannot be, and a blank one would shift every index after it.
			if strings.TrimSpace(turn.Content) == "" {
				continue
			}
			role, err := lmeRole(turn.Role)
			if err != nil {
				return Question{}, fmt.Errorf("eval: %s: %w", in.QuestionID, err)
			}
			if turn.HasAnswer {
				// The index this turn is about to take.
				q.Relevant = append(q.Relevant, len(q.Haystack))
			}
			q.Haystack = append(q.Haystack, mempher.AppendRequest{
				Content:    turn.Content,
				Role:       role,
				Source:     "longmemeval",
				OccurredAt: occurred,
				Binding:    mempher.Binding{"session": sessionID},
			})
		}
	}
	return q, nil
}

func lmeRole(role string) (mempher.Role, error) {
	switch role {
	case "user":
		return mempher.RoleUser, nil
	case "assistant":
		return mempher.RoleAssistant, nil
	default:
		return "", fmt.Errorf("unknown role %q: %w", role, mempher.ErrInvalidRole)
	}
}

// loose is a benchmark field that is usually a string and occasionally a bare
// number -- an answer of "2" is written 2. Decoding it as a string fails on
// those, and failing the whole run over a type the file is casual about would be
// the wrong kind of strict.
type loose string

func (l *loose) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		*l = loose(text)
		return nil
	}
	*l = loose(strings.Trim(string(data), `"`))
	return nil
}
