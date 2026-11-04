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
            "Co-Authored-By:  Devin AI <158243242+devin-ai-integration[bot]@users.noreply.github.com>\r\n"
            "Assisted-by: Example[bot] <bot@example.com>\r\n"
            "Co-authored-by: Example[bot] <bot@example.com>"
        )
        expected = (
            "Summary mentioning Co-authored-by: example[bot]\r\n\r\n"
            "Signed-off-by: Example[bot] <bot@example.com>\r\n"
            "Co-authored-by: Jane Bot <jane@example.com>\r\n"
            "Assisted-by:  Devin AI <158243242+devin-ai-integration[bot]@users.noreply.github.com>\r\n"
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
        accounts = {
            "user/123": {"type": "Bot"},
            "user/456": {"type": "User"},
            "users/Copilot": {"type": "Bot"},
            "users/alice": {"type": "User"},
        }
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

    @patch("squash_merge.api")
    def test_renamed_accounts_are_identified_by_id_even_if_old_names_are_reused(self, api):
        accounts = {
            "user/123": {"id": 123, "login": "new-bot", "type": "Bot"},
            "user/456": {"id": 456, "login": "new-alice", "type": "User"},
            "users/old-bot": {"id": 789, "login": "old-bot", "type": "User"},
            "users/old-alice": {"id": 987, "login": "old-alice", "type": "Bot"},
        }
        api.side_effect = lambda endpoint: accounts[endpoint]
        original = (
            "Co-authored-by: Robot <123+old-bot@users.noreply.github.com>\n"
            "Co-authored-by: Alice <456+old-alice@users.noreply.github.com>\n"
        )
        expected = (
            "Assisted-by: Robot <123+old-bot@users.noreply.github.com>\n"
            "Co-authored-by: Alice <456+old-alice@users.noreply.github.com>\n"
        )
        self.assertEqual(squash_merge.clean_message(original), expected)

    @patch("squash_merge.api")
    def test_renamed_human_with_unavailable_old_name_does_not_block_merge(self, api):
        identity = "Co-authored-by: Alice <456+old-alice@users.noreply.github.com>"

        def respond(endpoint, **kwargs):
            if endpoint == "graphql":
                return {
                    "data": {"repository": {"pullRequest": self.pr(viewerMergeBodyText=identity)}}
                }
            if endpoint == "user/456":
                return {"id": 456, "login": "new-alice", "type": "User"}
            if endpoint == "repos/owner/repo/pulls/42/merge":
                self.assertEqual(kwargs["payload"]["commit_message"], identity)
                return {"merged": True, "sha": "b" * 40}
            raise RuntimeError("gh: Not Found (HTTP 404)")

        api.side_effect = respond
        with contextlib.redirect_stdout(io.StringIO()) as output:
            squash_merge.run("owner/repo", 42, merge=True)
        self.assertIn("Merged: " + "b" * 40, output.getvalue())

    def test_empty_and_unrelated_messages_are_unchanged(self):
        for message in ("", "Fix a bug\n", "Co-authored-by:\nSigned-off-by: Person <a@b.com>\n"):
            with self.subTest(message=message):
                self.assertEqual(squash_merge.clean_message(message), message)

    def test_preserves_body_examples_and_changes_only_the_final_trailer_block(self):
        body = (
            "Document attribution syntax:\r\n\r\n"
            "```text\r\n"
            "Co-authored-by: Example[bot] <bot@example.com>\r\n"
            "```\r\n\r\n"
            "Co-authored-by: Example[bot] <bot@example.com>\r\n"
            "The line above is an example, not credit.\r\n\r\n"
        )
        original = (
            body + "Co-authored-by: Example[bot] <bot@example.com>\r\n"
            "Signed-off-by: Human <human@example.com>\r\n\r\n \t\r\n"
        )
        expected = (
            body + "Assisted-by: Example[bot] <bot@example.com>\r\n"
            "Signed-off-by: Human <human@example.com>\r\n\r\n \t\r\n"
        )
        self.assertEqual(squash_merge.clean_message(original), expected)
        self.assertEqual(squash_merge.clean_message(expected), expected)

    @patch("squash_merge.api", side_effect=AssertionError("Examples must not trigger lookups"))
    def test_body_examples_without_a_trailer_block_are_unchanged(self, api):
        for message in (
            "Example:\n\n```text\nCo-authored-by: Example <123+example@users.noreply.github.com>\n```\n",
            "Example:\n\nCo-authored-by: Example[bot] <bot@example.com>\nMore explanation.\n",
            "Here is an example:\nCo-authored-by: Example[bot] <bot@example.com>\n",
            "Here is an example:\n\n    Co-authored-by: Example[bot] <bot@example.com>\n",
        ):
            with self.subTest(message=message):
                self.assertEqual(squash_merge.clean_message(message), message)

    def test_trailer_continuations_are_preserved_without_preventing_bot_cleanup(self):
        original = (
            "Summary\n\n"
            "Co-authored-by: Robot[bot] <bot@example.com>\n"
            "Reviewed-by: Reviewer\n"
            " <reviewer@example.com>\n"
            "Notes: A quoted attribution line follows.\n"
            "\tCo-authored-by: Example[bot] <example@example.com>\n"
        )
        expected = (
            "Summary\n\n"
            "Assisted-by: Robot[bot] <bot@example.com>\n"
            "Reviewed-by: Reviewer\n"
            " <reviewer@example.com>\n"
            "Notes: A quoted attribution line follows.\n"
            "\tCo-authored-by: Example[bot] <example@example.com>\n"
        )
        self.assertEqual(squash_merge.clean_message(original), expected)

    def test_indented_coauthor_text_is_a_continuation_not_a_trailer_key(self):
        message = (
            "Summary\n\n"
            "Notes: A quoted attribution line follows.\n"
            "\tCo-authored-by: Example[bot] <example@example.com>\n"
        )
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
