# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep squash-message changes limited to bot co-author prefixes."""

import contextlib
import io
import unittest
from unittest.mock import patch

import squash_merge


class SquashMergeTest(unittest.TestCase):
    def test_changes_only_bot_coauthor_prefixes(self):
        original = (
            "Summary mentioning Co-authored-by: example[bot]\r\n\r\n"
            "Signed-off-by: Example[bot] <bot@example.com>\r\n"
            "Co-authored-by: Jane Bot <jane@example.com>\r\n"
            "\tCo-Authored-By:  Devin AI <158243242+devin-ai-integration[bot]@users.noreply.github.com>\r\n"
            "Assisted-by: Example[bot] <bot@example.com>\r\n"
            "Co-authored-by: Example[bot] <bot@example.com>"
        )
        expected = (
            "Summary mentioning Co-authored-by: example[bot]\r\n\r\n"
            "Signed-off-by: Example[bot] <bot@example.com>\r\n"
            "Co-authored-by: Jane Bot <jane@example.com>\r\n"
            "\tAssisted-by:  Devin AI <158243242+devin-ai-integration[bot]@users.noreply.github.com>\r\n"
            "Assisted-by: Example[bot] <bot@example.com>\r\n"
            "Assisted-by: Example[bot] <bot@example.com>"
        )
        self.assertEqual(squash_merge.clean_message(original), expected)
        self.assertEqual(squash_merge.clean_message(expected), expected)

    def test_recognizes_known_ai_service_addresses_without_guessing_from_names(self):
        original = (
            "Co-authored-by: Claude Sonnet <noreply@anthropic.com>\n"
            "Co-authored-by: Codex <noreply@openai.com>\n"
            "Co-authored-by: Claude <claude@example.com>\n"
        )
        expected = (
            "Assisted-by: Claude Sonnet <noreply@anthropic.com>\n"
            "Assisted-by: Codex <noreply@openai.com>\n"
            "Co-authored-by: Claude <claude@example.com>\n"
        )
        self.assertEqual(squash_merge.clean_message(original), expected)

    @patch("squash_merge.api")
    def test_looks_up_noreply_accounts_and_preserves_humans(self, api):
        accounts = {"users/Copilot": {"type": "Bot"}, "users/alice": {"type": "User"}}
        api.side_effect = lambda endpoint: accounts[endpoint]
        original = (
            "Co-authored-by: Copilot <123+Copilot@users.noreply.github.com>\n"
            "Co-authored-by: Alice <456+alice@users.noreply.github.com>\n"
            "Co-authored-by: Alice <alice@users.noreply.github.com>\n"
        )
        expected = (
            "Assisted-by: Copilot <123+Copilot@users.noreply.github.com>\n"
            "Co-authored-by: Alice <456+alice@users.noreply.github.com>\n"
            "Co-authored-by: Alice <alice@users.noreply.github.com>\n"
        )
        self.assertEqual(squash_merge.clean_message(original), expected)

    def test_empty_and_unrelated_messages_are_unchanged(self):
        for message in ("", "Fix a bug\n", "Co-authored-by:\nSigned-off-by: Person <a@b.com>\n"):
            with self.subTest(message=message):
                self.assertEqual(squash_merge.clean_message(message), message)

    def pr(self, **changes):
        return {
            "state": "OPEN",
            "isDraft": False,
            "isMergeQueueEnabled": False,
            "mergeStateStatus": "CLEAN",
            "headRefOid": "a" * 40,
            "viewerMergeHeadlineText": "Keep GitHub's title (#42)",
            "viewerMergeBodyText": (
                "Keep GitHub's body.\n\n"
                "Co-authored-by: Robot[bot] <robot@example.com>\n"
                "Signed-off-by: Human <human@example.com>"
            ),
            **changes,
        }

    @patch("squash_merge.api")
    def test_preview_does_not_merge(self, api):
        api.return_value = {"data": {"repository": {"pullRequest": self.pr()}}}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            squash_merge.run("owner/repo", 42, merge=False)
        self.assertEqual(api.call_count, 1)
        self.assertIn("Assisted-by: Robot[bot]", output.getvalue())

    @patch("squash_merge.api")
    def test_merge_submits_generated_message_and_expected_head(self, api):
        api.side_effect = [
            {"data": {"repository": {"pullRequest": self.pr()}}},
            {"merged": True, "sha": "b" * 40},
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            squash_merge.run("owner/repo", 42, merge=True)
        api.assert_called_with(
            "repos/owner/repo/pulls/42/merge",
            method="PUT",
            payload={
                "merge_method": "squash",
                "sha": "a" * 40,
                "commit_title": "Keep GitHub's title (#42)",
                "commit_message": (
                    "Keep GitHub's body.\n\n"
                    "Assisted-by: Robot[bot] <robot@example.com>\n"
                    "Signed-off-by: Human <human@example.com>"
                ),
            },
        )

    @patch("squash_merge.api")
    def test_merge_does_not_log_pr_content_as_workflow_commands(self, api):
        api.side_effect = [
            {
                "data": {
                    "repository": {
                        "pullRequest": self.pr(
                            viewerMergeBodyText="::error::Text from a PR description"
                        )
                    }
                }
            },
            {"merged": True, "sha": "b" * 40},
        ]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            squash_merge.run("owner/repo", 42, merge=True)
        self.assertNotIn("::error::", output.getvalue())
        self.assertIn("b" * 40, output.getvalue())

    @patch("squash_merge.api")
    def test_refuses_unready_or_queued_merges(self, api):
        for changes in (
            {"state": "CLOSED"},
            {"state": "MERGED"},
            {"isDraft": True},
            {"isMergeQueueEnabled": True},
            {"mergeStateStatus": "BLOCKED"},
            {"mergeStateStatus": "UNKNOWN"},
            {"mergeStateStatus": "UNSTABLE"},
        ):
            with self.subTest(changes=changes):
                api.reset_mock()
                api.return_value = {"data": {"repository": {"pullRequest": self.pr(**changes)}}}
                with self.assertRaises(RuntimeError):
                    squash_merge.run("owner/repo", 42, merge=True)
                self.assertEqual(api.call_count, 1)

    @patch("squash_merge.api")
    def test_failed_bot_lookup_prevents_merge(self, api):
        api.side_effect = [
            {
                "data": {
                    "repository": {
                        "pullRequest": self.pr(
                            viewerMergeBodyText="Co-authored-by: Unknown <123+unknown@users.noreply.github.com>"
                        )
                    }
                }
            },
            RuntimeError("GitHub is unavailable"),
        ]
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            squash_merge.run("owner/repo", 42, merge=True)
        self.assertEqual(api.call_count, 2)


if __name__ == "__main__":
    unittest.main()
