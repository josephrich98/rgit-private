---
name: send-email-when-finished
description: After completing any task, send a notification email to the user. Triggers whenever the user's prompt ends with the phrase "send email when finished". Do NOT trigger on any other phrasing.
---

# Send Email When Finished

When the user's prompt ends with "send email when finished", complete the requested task normally, then send a notification email at the very end.

## Instructions

1. Identify and complete the user's actual task (everything in the prompt before "send email when finished").
2. After the task is fully done, run this shell command exactly:

```bash
EMAIL_APP_PASSWORD="$EMAIL_APP_PASSWORD" ~/bin/send_email.py claude
```

3. Do not announce that you are about to send the email beforehand. Just send it after the work is done and confirm it was sent in one short line, e.g. "Email sent."

## Notes

- The email body will be the single word "claude" (passed as an argument to the script).
- The script reads `EMAIL_APP_PASSWORD` from the environment — it is already set in the user's shell.
- Do not modify the subject, body, or recipients.
- If the script fails, report the error to the user.
