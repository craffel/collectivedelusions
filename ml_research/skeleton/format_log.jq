def truncate($n):
  if (length > $n)
  then .[0:$n] + "\n... [+\(length - $n) chars truncated]"
  else . end;

def indent($prefix):
  split("\n") | map($prefix + .) | join("\n");

def result_text:
  if type == "string" then .
  elif type == "array" then map(.text // "") | join("\n")
  else tostring end;

def render_tool($name; $input):
  if $name == "TodoWrite" then
    "[TodoWrite]\n" +
    (($input.todos // []) | map(
      (if .status == "completed" then "  [x] "
       elif .status == "in_progress" then "  [~] "
       else "  [ ] " end) + .content
    ) | join("\n"))

  elif $name == "Bash" then
    "[Bash]" +
    (if $input.description then "  # " + $input.description else "" end) +
    "\n  $ " + (($input.command // "") | indent("    ") | .[4:])

  elif $name == "Write" then
    "[Write] " + ($input.file_path // "") + "\n" +
    "  ---\n" +
    (($input.content // "") | indent("  | ")) + "\n" +
    "  ---"

  elif $name == "Edit" then
    "[Edit] " + ($input.file_path // "") +
    (if $input.replace_all then "  (replace_all)" else "" end) + "\n" +
    "  --- old\n" +
    (($input.old_string // "") | indent("  - ")) + "\n" +
    "  +++ new\n" +
    (($input.new_string // "") | indent("  + "))

  elif $name == "Read" then
    "[Read] " + ($input.file_path // "") +
    (if ($input.offset // $input.limit // $input.pages)
     then "  (" +
       ([if $input.offset then "offset=\($input.offset)" else empty end,
         if $input.limit then "limit=\($input.limit)" else empty end,
         if $input.pages then "pages=\($input.pages)" else empty end]
        | join(", ")) + ")"
     else "" end)

  elif $name == "Glob" then
    "[Glob] \($input.pattern // "")" +
    (if $input.path then " in \($input.path)" else "" end)

  elif $name == "Grep" then
    "[Grep] \($input.pattern // "")" +
    (if $input.path then " in \($input.path)" else "" end) +
    (if $input.glob then "  glob=\($input.glob)" else "" end) +
    (if $input.type then "  type=\($input.type)" else "" end) +
    (if $input.output_mode then "  mode=\($input.output_mode)" else "" end)

  elif $name == "WebFetch" or $name == "WebSearch" then
    "[\($name)] " + ($input.url // $input.query // "")

  elif $name == "ScheduleWakeup" then
    "[ScheduleWakeup] in \($input.delaySeconds)s — \($input.reason // "")"

  elif $name == "ToolSearch" then
    "[ToolSearch] \($input.query // "")"

  else
    "[\($name)]\n" +
    ($input | to_entries | map("  \(.key): \(.value | tostring | .[0:300])") | join("\n"))
  end;

if .type == "stream_event" then
  .event as $ev |
  if $ev.type == "content_block_start" and $ev.content_block.type == "text" then
    "\n\n--- Claude ---\n"
  elif $ev.type == "content_block_delta" and $ev.delta.type == "text_delta" then
    $ev.delta.text
  else "" end

elif .type == "assistant" then
  ((.message.content // []) | map(
    if .type == "tool_use" then "\n\n" + render_tool(.name; .input)
    else "" end
  ) | join(""))

elif .type == "user" then
  ((.message.content // []) | map(
    if .type == "tool_result" then
      "\n" + (if .is_error then "  !! " else "  -> " end) +
      "result:\n" +
      ((.content | result_text | truncate(1200)) | indent("    "))
    else "" end
  ) | join(""))

elif .type == "result" then
  "\n\n=== FINAL ===\n" + (.result // "") + "\n"

else "" end
