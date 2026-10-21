## Use case: medicine_before_food (reminder to take a tablet shortly BEFORE a meal)
The agent said: "It is time for your <medicine>. Please take it now, about 15 minutes before your <meal>. Have you taken it?"
step_kind "yes_no", `topic` is the medicine name.

- yes_no true: they state they took it (now or just before this call).
- yes_no false: not yet, will take later, forgot. Intent stays "answer".
- "Hold on, let me get it" / "I forgot, I'll take it right now" / "going to get it": they are acting now. Intent "wait"
  (do not log anything; they will report back).
- "I forgot! ... okay done" in one reply: the final state matters (taken), quote the part that says it.
- They have already eaten, or the timing seems wrong, or they ask whether to take it: intent "medical_question".
- They are unsure WHICH tablet it is ("the white one", "I have many") or do not name it: intent "unclear";
  do not treat "I took a tablet" as confirmation for this specific medicine.
- Refusal or side effects as a reason not to take it ("makes me sick, I won't"): intent "refusal"
  (fill mild_symptom if they describe one).
- Taken by someone else / about someone else is not an answer: "unclear".

Examples:
"Yes, I've taken it just now" -> {"intent":"answer","yes_no":true,"evidence":"I've taken it just now","confidence":0.95}
"Oh, I forgot, hold on" -> {"intent":"wait","evidence":"hold on","confidence":0.85}
"Oh I had forgotten, I'll go get it now" -> {"intent":"wait","evidence":"I'll go get it now","confidence":0.85}
"Not yet, I'll take it after the news" -> {"intent":"answer","yes_no":false,"evidence":"Not yet","confidence":0.9}
"I already ate lunch early. Should I still take it?" -> {"intent":"medical_question","evidence":"Should I still take it","confidence":0.9}
"I think I took it, not sure" -> {"intent":"unsure","evidence":"not sure","confidence":0.85}
"Which tablet is this? I take so many, I took the white one" -> {"intent":"unclear","evidence":"","confidence":0.3,"reason":"cannot tell which tablet"}
"I don't want to take it, it makes me feel sick" -> {"intent":"refusal","mild_symptom":"feels sick from it","evidence":"I don't want to take it","confidence":0.9}
