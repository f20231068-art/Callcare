## Use case: meal_discovery (first call, learning usual meal times)
The agent asked: "What time do you usually have <meal>?" The meal is in `topic`. step_kind is "time".

Clear answer: a specific time, converted to 24h "HH:MM". Use the meal to resolve am/pm: breakfast is morning,
lunch around noon, snack afternoon/evening, dinner evening/night.
Not clear (intent "unclear"): "whenever I feel hungry", "depends", "no fixed time", "after morning prayers".
Do not guess a time.
Ranges: "between one and two" => start of the range, "13:00", confidence 0.6.
"Half past eight" = 08:30, "quarter to nine" = 08:45, "saade aath" (Hindi) = 08:30.

Examples (reply -> output):
"around 8" (breakfast) -> {"intent":"answer","time":"08:00","evidence":"around 8","confidence":0.9}
"one thirty or so" (lunch) -> {"intent":"answer","time":"13:30","evidence":"one thirty","confidence":0.9}
"whenever I wake up" (breakfast) -> {"intent":"unclear","evidence":"","confidence":0.3,"reason":"no specific time"}
"Who is this? Why do you want to know?" -> {"intent":"identity","evidence":"Who is this","confidence":0.9}
"I have tea first, then idli at half past eight, my son sends the maid" (breakfast) ->
  {"intent":"answer","time":"08:30","evidence":"half past eight","confidence":0.85}
