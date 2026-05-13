#!/bin/bash
#SBATCH --job-name=autoresearch-agent
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88
#SBATCH --gpus-per-node=8
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=6:00:00

echo "Starting Autoresearch Agent Job: $SLURM_JOB_ID"
echo "Allocated Node: $(hostname)"

# Repeatedly invoke the agent with a 10-minute (600 seconds) delay
while true; do
    echo "[$(date)] Invoking claude..."
    
    # Run the agent autonomously
    /admin/home/craffel/.local/bin/claude --bare --output-format stream-json --verbose --include-partial-messages --dangerously-skip-permissions --model "claude-opus-4-7" -p "Follow the instructions in plan.md" | jq -rj '
  if .type == "system" then
    "\n[\(.subtype | ascii_upcase): \(.status)]\n"
  elif .type == "stream_event" then
    .event |
    if .type == "content_block_start" then
      .content_block |
      if .type == "thinking" then "\n[THINKING]\n"
      elif .type == "tool_use" then "\n[TOOL: \(.name)]\n"
      elif .type == "text" then "\n[RESPONSE]\n"
      else ""
      end
    elif .type == "content_block_delta" then
      .delta |
      if .type == "thinking_delta" then .thinking
      elif .type == "text_delta" then .text
      elif .type == "input_json_delta" then .partial_json
      else ""
      end
    elif .type == "content_block_stop" then "\n"
    else ""
    end
  elif .type == "result" then
    "\n[RESULT]\n\(.result)\n"
  else ""
  end
'
    
    echo "[$(date)] claude-cli invocation finished. Sleeping for 10 minutes."
    sleep 600
done
