# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Squash a PR with bot co-authors recorded as assistants."""

import argparse
import json
import os
import re
import subprocess
import sys
from functools import cache

PR_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      state isDraft headRefOid isMergeQueueEnabled mergeStateStatus
      viewerMergeHeadlineText(mergeType: SQUASH)
      viewerMergeBodyText(mergeType: SQUASH)
    }
  }
}
"""
AI_EMAILS = {"noreply@anthropic.com", "noreply@openai.com"}


def api(endpoint, *, method="GET", payload=None):
    """Use gh's authentication; pass JSON as data, never as shell commands."""
    command = ["gh", "api", "--method", method, endpoint]
    if payload is not None:
        command += ["--input", "-"]
    result = subprocess.run(
        command,
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    response = json.loads(result.stdout)
    if isinstance(response, dict) and response.get("errors"):
        raise RuntimeError(str(response["errors"]))
    return response


def clean_message(message):
    """Replace bot co-author keys in the final trailer paragraph."""
    lines = message.splitlines(keepends=True)
    end = len(lines)
    while end and not lines[end - 1].strip():
        end -= 1
    start = end
    while start and lines[start - 1].strip():
        start -= 1
    trailers = lines[start:end]
    # A trailer key starts at column zero; indented lines continue its value.
    if (
        not trailers
        or not re.match(r"[^\s:]+:", trailers[0])
        or any(not re.match(r"(?:[^\s:]+:|[ \t])", line) for line in trailers[1:])
    ):
        return message

    @cache
    def is_bot(identity):
        if "[bot]" in identity.lower():
            return True
        email = re.search(r"<([^<>\s]+)>[ \t]*$", identity)
        if not email:
            return False
        if email[1].lower() in AI_EMAILS:
            return True
        account = re.fullmatch(
            r"(?:(\d+)\+)?([a-z0-9][a-z0-9-]*)@users\.noreply\.github\.com",
            email[1],
            re.IGNORECASE,
        )
        if not account:
            return False
        endpoint = f"user/{account[1]}" if account[1] else f"users/{account[2]}"
        return api(endpoint)["type"] == "Bot"

    def replace(match):
        if is_bot(match[1]):
            return f"Assisted-by:{match[1]}"
        return match[0]

    cleaned = re.sub(
        r"^Co-authored-by:([^\r\n]*)",
        replace,
        "".join(trailers),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return "".join(lines[:start]) + cleaned + "".join(lines[end:])


def run(repository, number, *, merge=False):
    owner, name = repository.split("/")
    response = api(
        "graphql",
        method="POST",
        payload={"query": PR_QUERY, "variables": {"owner": owner, "name": name, "number": number}},
    )
    pr = response["data"]["repository"]["pullRequest"]
    if not pr:
        raise RuntimeError("Pull request not found.")
    if merge:
        if pr["state"] != "OPEN" or pr["isDraft"]:
            raise RuntimeError("Only open, ready-for-review pull requests can be merged.")
        if pr["isMergeQueueEnabled"]:
            raise RuntimeError("Merge queues do not support this custom squash message.")
        if pr["mergeStateStatus"] != "CLEAN":
            raise RuntimeError(f"Pull request is not ready to merge ({pr['mergeStateStatus']}).")

    title = pr["viewerMergeHeadlineText"]
    body = clean_message(pr["viewerMergeBodyText"])
    if not merge:
        print(f"{title}\n\n{body}")
    if merge:
        result = api(
            f"repos/{repository}/pulls/{number}/merge",
            method="PUT",
            payload={
                "merge_method": "squash",
                "sha": pr["headRefOid"],
                "commit_title": title,
                "commit_message": body,
            },
        )
        if not result.get("merged"):
            raise RuntimeError(result.get("message", "GitHub refused the merge."))
        print(f"\nMerged: {result['sha']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", default=os.environ.get("GITHUB_REPOSITORY"), help="Repository in owner/name form"
    )
    parser.add_argument("--pr", type=int, required=True, help="Pull request number")
    parser.add_argument(
        "--merge", action="store_true", help="Perform the merge; otherwise preview only"
    )
    args = parser.parse_args()
    if not args.repo or not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo) or args.pr <= 0:
        parser.error("Provide --repo owner/name and a positive --pr number.")
    try:
        run(args.repo, args.pr, merge=args.merge)
    except (RuntimeError, OSError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
