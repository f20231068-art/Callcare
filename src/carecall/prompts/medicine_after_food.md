## Use case: medicine_after_food (reminder to take a tablet after a meal)
The agent said: "It is time for your <medicine>. Please take it now, after your <meal>. Have you taken it?"
step_kind "yes_no", `topic` is the medicine name.

- yes_no true: they took it (now, or earlier today when they ate). If they say they took it EARLIER than this call and
  ask whether to take another / extra: intent "medical_question" (possible double dose), quote the question.
- yes_no false: not yet, later, sleepy, forgot.
- "Let me get it" / "I'll take it right now" / "one minute": they are acting now. Intent "wait" (log nothing).
- Mixed-language replies are common: "haan le liya" = yes, taken; "abhi nahi" = not yet; "kha liya aur goli bhi le li"
  = ate and took the tablet (true). Understand them; quote them exactly.
- Distracted replies ("hmm? what? the match is on"): intent "repeat" if they ask to hear again, else "off_topic".
- Not about the person ("my wife took hers"): "unclear".

Examples:
"Yes madam I took it after dinner" -> {"intent":"answer","yes_no":true,"evidence":"I took it after dinner","confidence":0.95}
"Not yet, very sleepy, I will take it later" -> {"intent":"answer","yes_no":false,"evidence":"Not yet","confidence":0.9}
"haan le liya, abhi khana khaya aur goli le li" -> {"intent":"answer","yes_no":true,"evidence":"goli le li","confidence":0.9}
"I took it already in the evening, shall I take another one now?" -> {"intent":"medical_question","evidence":"shall I take another one","confidence":0.9}
"Hmm? Oh the cricket is on, what did you say?" -> {"intent":"repeat","evidence":"what did you say","confidence":0.8}
