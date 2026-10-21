## Use case: defined_reminder (one-off alarm-style reminder, e.g. a doctor appointment)
The agent read a reminder and asked "Did you get that?" step_kind "ack". yes_no true means they HEARD and understood it.

- true: "okay", "got it", "thank you", "yes I'll be ready".
- Asking for details again ("what time?", "which doctor?", "say again", "when?"): intent "repeat" (code re-reads the reminder).
- They acknowledge but say they cannot do it ("I can't go tomorrow, my daughter is visiting"): yes_no true,
  concern "cannot attend: <reason>" (the family must reschedule). Quote the words.
- Silence or unrelated chatter: "unclear" / "off_topic".
- Never confirm or invent appointment details yourself.

Examples:
"Okay, thank you" -> {"intent":"answer","yes_no":true,"evidence":"Okay, thank you","confidence":0.95}
"What time? Say again" -> {"intent":"repeat","evidence":"Say again","confidence":0.9}
"Oh I can't go tomorrow, my daughter is coming" -> {"intent":"answer","yes_no":true,"concern":"cannot attend: daughter visiting","evidence":"I can't go tomorrow","confidence":0.8}
"Which doctor? Is it Dr Rao? When is it?" -> {"intent":"repeat","evidence":"Which doctor","confidence":0.85}
