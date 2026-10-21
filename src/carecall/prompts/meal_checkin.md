## Use case: meal_checkin (did they eat a given meal, and what)
step_kind "yes_no": agent asked "Have you had your <meal> today?". step_kind "items": agent asked "What did you have?".

yes_no:
- true only if the person says they ate THIS meal. "I had some tea" is not lunch.
  If the same reply already names what they ate ("yes, dal and rice"), also fill "items" so we do not ask again.
- false + skipped false: "not yet", "will eat later", "haven't eaten" (they intend to eat).
- false + skipped true: deliberately not eating ("skipping lunch", "not eating today").
- "I don't feel like eating", "not hungry", "no appetite": yes_no false, skipped true, concern "low appetite".
- Unsure ("did I? I don't remember", "maybe"): intent "unsure". NEVER choose true or false for them.
- Another person's meal ("my husband ate") is not an answer: "unclear".
items: foods they say they ate, short lowercase nouns, in the language used (transliterate if needed).
  Include only foods, not quantities or feelings. If they list nothing specific ("just a little"), intent "unclear".

Examples:
"Yes I had lunch" (yes_no) -> {"intent":"answer","yes_no":true,"evidence":"Yes I had lunch","confidence":0.95}
"Not yet, watching my serial, will eat later" -> {"intent":"answer","yes_no":false,"skipped":false,"evidence":"Not yet","confidence":0.9}
"No. Nobody to eat with, I don't feel like it" -> {"intent":"answer","yes_no":false,"skipped":true,"concern":"low appetite, eating alone","evidence":"I don't feel like it","confidence":0.8}
"Yes but I felt a bit dizzy after" -> {"intent":"answer","yes_no":true,"mild_symptom":"a bit dizzy","evidence":"Yes","confidence":0.9}
"Did I? I don't remember, maybe" -> {"intent":"unsure","evidence":"I don't remember","confidence":0.9}
"rice, sambar and a little curd" (items) -> {"intent":"answer","items":["rice","sambar","curd"],"evidence":"rice, sambar and a little curd","confidence":0.9}
