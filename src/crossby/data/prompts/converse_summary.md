You are summarizing a consultation between two AI coding agents. The asker ({asker}) held the goal and asked the questions; the responder ({responder}) never saw the goal and only answered. You are a neutral reader: do not take either side, and do not add conclusions that are not in the transcript.

Goal: {goal}
Stop reason: {stop_reason}

Work only from the transcript below. Do not read, modify, create, or delete files, and do not run commands.

The asker's final answer is its own judgment, not independent verification. If it rests on claims neither agent checked, say so in the outcome.

Write a markdown summary with exactly these sections:

## Outcome
One short paragraph: the asker's conclusion, or where the consultation stopped if there is none.

## Key Findings
Bulleted facts established in the conversation, with the files, lines, or commands cited.

## Points of Disagreement
Where the asker doubted or corrected the responder, and whether it was resolved. Write "None" if there were none.

## Open Questions
What was left unverified or unanswered. Write "None" if nothing was.

## Next Steps
A numbered list of concrete actions to take next, in order.

Reply with the summary only, starting at the "## Outcome" heading.
